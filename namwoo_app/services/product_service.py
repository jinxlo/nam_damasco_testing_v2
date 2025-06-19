# NAMWOO/services/product_service.py

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation as InvalidDecimalOperation
from datetime import datetime

import numpy as np
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..models.product import Product
from ..utils import db_utils, embedding_utils, text_utils
from ..config import Config

logger = logging.getLogger(__name__)

# --- Semantic Helpers ---
_KNOWN_COLORS = {
    'negro', 'blanco', 'azul', 'rojo', 'verde', 'gris', 'plata', 'dorado',
    'rosado', 'violeta', 'morado', 'amarillo', 'naranja', 'marrón', 'beige',
    'celeste', 'turquesa', 'lila', 'crema', 'grafito', 'titanio', 'cobre',
    'negra', 'blanca', 'claro', 'oscuro', 'marino'
}
_SKU_PAT = re.compile(r'\b(SM-[A-Z0-9]+[A-Z]*|[A-Z0-9]{8,})\b')

def _extract_base_name_and_color(item_name: str) -> Tuple[str, Optional[str]]:
    """
    Intelligently splits an item name into a "base name" and a "color" by
    identifying and removing known color words and SKU-like suffixes.
    """
    if not item_name: return "", None
    name_without_sku = _SKU_PAT.sub('', item_name).strip()
    words = name_without_sku.split()
    base_name_parts, color_parts = [], []
    color_found = False
    for i in range(len(words) - 1, -1, -1):
        word = words[i]
        if not color_found and word.lower() in _KNOWN_COLORS:
            color_parts.insert(0, word)
        else:
            color_found = True
            base_name_parts.insert(0, word)
    final_base_name = " ".join(base_name_parts).strip()
    final_color = " ".join(color_parts).strip()
    if not final_color: return name_without_sku, None
    return final_base_name, final_color.capitalize()

# --- Search Products ---
def search_local_products(
    query_text: str,
    limit: int = 300, 
    filter_stock: bool = True,
    min_score: float = 0.10,
    warehouse_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Performs a vector search for products, then groups the results by a "base name"
    (stripping color variants). This provides a clean, de-duplicated list of core products,
    each with a nested list of available colors/prices and all locations where it is available.
    """
    if not query_text or not isinstance(query_text, str):
        logger.warning("Search query is empty or invalid.")
        return {}

    logger.info(
        "Vector search initiated: '%s…' (fetch_limit=%d, stock=%s, min_score=%.2f)",
        query_text[:80], limit, filter_stock, min_score
    )
    embedding_model = Config.OPENAI_EMBEDDING_MODEL if hasattr(Config, 'OPENAI_EMBEDDING_MODEL') else "text-embedding-3-small"
    query_emb = embedding_utils.get_embedding(query_text, model=embedding_model)
    if not query_emb:
        logger.error("Query embedding generation failed – aborting search.")
        return None

    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for search.")
            return None
        try:
            q = session.query(Product, (1 - Product.embedding.cosine_distance(query_emb)).label("similarity"))
            
            # This filter ensures we only ever fetch items with stock > 0
            if filter_stock:
                q = q.filter(Product.stock > 0)
            
            if warehouse_names:
                q = q.filter(Product.warehouse_name.in_(warehouse_names))
                
            q = q.filter(Product.item_group_name == "DAMASCO TECNO")
            q = q.filter((1 - Product.embedding.cosine_distance(query_emb)) >= min_score)
            q = q.order_by(Product.embedding.cosine_distance(query_emb)).limit(limit)
            raw_db_rows: List[Tuple[Product, float]] = q.all()

            # The dictionary structure handles de-duplication automatically.
            grouped_products: Dict[str, Dict[str, Any]] = {}

            for product_entry, sim_score in raw_db_rows:
                base_name, color = _extract_base_name_and_color(product_entry.item_name)
                if not base_name: continue

                # If we see a base_name for the first time, create its master record.
                if base_name not in grouped_products:
                    description_text = product_entry.llm_summarized_description or text_utils.strip_html_to_text(product_entry.description or "")
                    grouped_products[base_name] = {
                        "base_name": base_name, "brand": product_entry.brand, "category": product_entry.category,
                        "sub_category": product_entry.sub_category, "description": description_text.strip(),
                        "variants": [], "locations": []
                    }
                
                # Add a new variant if we haven't seen this exact color/price combo before for this base product.
                variant_exists = any(
                    v['full_item_name'] == product_entry.item_name for v in grouped_products[base_name]['variants']
                )
                if not variant_exists:
                    grouped_products[base_name]['variants'].append({
                        "color": color or "Color no especificado",
                        "price": float(product_entry.price) if product_entry.price is not None else None,
                        "price_bolivar": float(product_entry.price_bolivar) if product_entry.price_bolivar is not None else None,
                        "full_item_name": product_entry.item_name,
                        "item_code": product_entry.item_code
                    })

                # Always add the location information for this specific raw entry.
                grouped_products[base_name]["locations"].append({
                    "warehouse_name": product_entry.warehouse_name,
                    "branch_name": product_entry.branch_name,
                    "stock": product_entry.stock,
                    "color_specific_item_name": product_entry.item_name
                })

            final_product_list = list(grouped_products.values())
            
            logger.info(
                "Vector search found %d raw DB entries, grouped into %d unique base products.",
                len(raw_db_rows), len(final_product_list)
            )

            return {"status": "success", "products_grouped": final_product_list}
        except SQLAlchemyError as db_exc:
            logger.exception("Database error during product search: %s", db_exc)
            return None
        except Exception as exc:
            logger.exception("Unexpected error during product search: %s", exc)
            return None

def _normalize_string(value: Any) -> Optional[str]:
    if value is None: return None
    s = str(value).strip()
    return s if s else None

def get_product_by_id_from_db(db_session: Session, product_id: str) -> Optional[Product]:
    if not product_id: return None
    return db_session.query(Product).filter(Product.id == product_id).first()

def add_or_update_product_in_db(
    session: Session,
    damasco_product_data_camel: Dict[str, Any],
    embedding_vector: Optional[Any],
    text_used_for_embedding: Optional[str],
    llm_summarized_description_to_store: Optional[str]
) -> Tuple[bool, str]:
    item_code = _normalize_string(damasco_product_data_camel.get("itemCode"))
    whs_name = _normalize_string(damasco_product_data_camel.get("whsName"))
    _temp_product_id_for_upsert: Optional[str] = None
    if item_code and whs_name:
        sanitized_whs_name = re.sub(r'[^a-zA-Z0-9_-]', '_', whs_name)
        _temp_product_id_for_upsert = f"{item_code}_{sanitized_whs_name}"
        if len(_temp_product_id_for_upsert) > 512:
            _temp_product_id_for_upsert = _temp_product_id_for_upsert[:512]
    product_id_to_upsert = _temp_product_id_for_upsert
    if not product_id_to_upsert:
        logger.error("Could not derive product_id_to_upsert from itemCode/whsName.")
        return False, "Missing product_id_to_upsert (could not derive)."
    if not damasco_product_data_camel or not isinstance(damasco_product_data_camel, dict):
        return False, "Missing or invalid Damasco product data (camelCase)."
    embedding_vector_for_db: Optional[List[float]] = None
    if embedding_vector is not None:
        if isinstance(embedding_vector, np.ndarray):
            embedding_vector_for_db = embedding_vector.tolist()
        elif isinstance(embedding_vector, list):
            embedding_vector_for_db = embedding_vector
        else:
            logger.error(f"Product ID {product_id_to_upsert}: Unexpected embedding vector type ({type(embedding_vector)}).")
            return False, f"Invalid embedding vector type for {product_id_to_upsert}."
        expected_dim = Config.EMBEDDING_DIMENSION if hasattr(Config, 'EMBEDDING_DIMENSION') else None
        if expected_dim and embedding_vector_for_db and len(embedding_vector_for_db) != expected_dim:
            return False, (f"Embedding dimension mismatch for {product_id_to_upsert} (expected {expected_dim}, got {len(embedding_vector_for_db)}).")
    if embedding_vector_for_db is not None and not text_used_for_embedding:
        logger.warning(f"Product ID {product_id_to_upsert}: Embedding vector present, but text_used_for_embedding is missing.")
    log_prefix = f"ProductService DB Upsert (ID='{product_id_to_upsert}'):"
    raw_html_description_to_store = damasco_product_data_camel.get("description")
    item_name = _normalize_string(damasco_product_data_camel.get("itemName"))
    price_from_damasco = damasco_product_data_camel.get("price")
    normalized_price_for_db: Optional[Decimal] = None
    if price_from_damasco is not None:
        try:
            normalized_price_for_db = Decimal(str(price_from_damasco)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        except InvalidDecimalOperation:
            logger.warning(f"{log_prefix} Invalid price value '{price_from_damasco}', treating as None.")
    price_bolivar_from_damasco = damasco_product_data_camel.get("priceBolivar")
    normalized_price_bolivar_for_db: Optional[Decimal] = None
    if price_bolivar_from_damasco is not None:
        try:
            normalized_price_bolivar_for_db = Decimal(str(price_bolivar_from_damasco)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        except InvalidDecimalOperation:
            logger.warning(f"{log_prefix} Invalid priceBolivar value '{price_bolivar_from_damasco}', treating as None.")
    stock_from_damasco = damasco_product_data_camel.get("stock")
    normalized_stock_for_db = 0
    if stock_from_damasco is not None:
        try:
            normalized_stock_for_db = int(stock_from_damasco)
            if normalized_stock_for_db < 0:
                normalized_stock_for_db = 0
        except (ValueError, TypeError):
            logger.warning(f"{log_prefix} Invalid stock value '{stock_from_damasco}', defaulting to 0.")
    norm_raw_html = _normalize_string(raw_html_description_to_store)
    norm_llm_summary = _normalize_string(llm_summarized_description_to_store)
    norm_searchable_text = _normalize_string(text_used_for_embedding)
    new_values_map = {
        "item_code": item_code, "item_name": item_name, "description": norm_raw_html,
        "llm_summarized_description": norm_llm_summary,
        "category": _normalize_string(damasco_product_data_camel.get("category")),
        "sub_category": _normalize_string(damasco_product_data_camel.get("subCategory")),
        "brand": _normalize_string(damasco_product_data_camel.get("brand")),
        "line": _normalize_string(damasco_product_data_camel.get("line")),
        "item_group_name": _normalize_string(damasco_product_data_camel.get("itemGroupName")),
        "warehouse_name": whs_name, "branch_name": _normalize_string(damasco_product_data_camel.get("branchName")),
        "price": normalized_price_for_db, "price_bolivar": normalized_price_bolivar_for_db,
        "stock": normalized_stock_for_db, "searchable_text_content": norm_searchable_text,
        "embedding": embedding_vector_for_db, "source_data_json": damasco_product_data_camel
    }
    try:
        entry = get_product_by_id_from_db(session, product_id_to_upsert)
        if entry:
            needs_update, changed_fields_log_details = False, []
            for field_key, new_value in new_values_map.items():
                if field_key == "source_data_json":
                    if entry.source_data_json != new_value:
                        needs_update = True
                        changed_fields_log_details.append(f"{field_key}: (JSON content changed)")
                    continue
                db_value, is_different = getattr(entry, field_key, None), False
                if field_key == "embedding":
                    db_value_list = db_value.tolist() if isinstance(db_value, np.ndarray) else db_value
                    if (db_value_list is None) != (new_value is None) or \
                       (db_value_list and not np.array_equal(np.array(db_value_list, dtype=float), np.array(new_value, dtype=float))):
                        is_different = True
                elif field_key in ["price", "price_bolivar"]:
                    if (db_value is None) != (new_value is None) or (db_value is not None and db_value != new_value):
                        is_different = True
                elif db_value != new_value:
                    is_different = True
                if is_different:
                    needs_update, log_new_val, log_db_val = True, str(new_value), str(db_value)
                    if len(log_new_val) > 70: log_new_val = log_new_val[:70] + "..."
                    if len(log_db_val) > 70: log_db_val = log_db_val[:70] + "..."
                    changed_fields_log_details.append(f"{field_key}: (DB='{log_db_val}' -> New='{log_new_val}')")
            if not needs_update:
                logger.info(f"{log_prefix} No changes detected. Skipping DB write.")
                return True, "skipped_no_change"
            logger.info(f"{log_prefix} Changes detected. Updating entry. Details: {'; '.join(changed_fields_log_details)}")
            for key, value in new_values_map.items():
                setattr(entry, key, value)
            entry.updated_at = datetime.utcnow()
            session.commit()
            op_msg = f"updated (Changes: {', '.join(changed_fields_log_details[:3])}" + (" ...)" if len(changed_fields_log_details) > 3 else ")")
            return True, op_msg
        else:
            logger.info(f"{log_prefix} New product. Adding to DB.")
            entry = Product(id=product_id_to_upsert, **new_values_map)
            session.add(entry)
            session.commit()
            return True, "added_new"
    except SQLAlchemyError as db_exc:
        session.rollback()
        logger.error(f"{log_prefix} DB error during add/update: {db_exc}", exc_info=True)
        err_str = str(db_exc).lower()
        if "violates unique constraint" in err_str: return False, f"db_constraint_violation: {str(db_exc)[:200]}"
        return False, f"db_sqlalchemy_error: {str(db_exc)[:200]}"
    except Exception as exc: 
        session.rollback()
        logger.exception(f"{log_prefix} Unexpected error processing: {exc}")
        return False, f"db_unexpected_error: {str(exc)[:200]}"

# --- Getter functions ---
def get_live_product_details_by_sku(item_code_query: str) -> Optional[List[Dict[str, Any]]]:
    if not (normalized_item_code := _normalize_string(item_code_query)):
        logger.warning(f"get_live_product_details_by_sku: item_code_query '{item_code_query}' is invalid.")
        return []
    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for get_live_product_details_by_sku.")
            return None
        try:
            product_entries = session.query(Product).filter_by(item_code=normalized_item_code).all()
            if not product_entries:
                logger.info(f"No product entries found with item_code: {normalized_item_code}")
                return []
            results = [entry.to_dict() for entry in product_entries]
            logger.info(f"Found {len(results)} locations for item_code: {normalized_item_code}")
            return results
        except SQLAlchemyError as db_exc:
            logger.exception(f"DB error fetching product by item_code: {normalized_item_code}, Error: {db_exc}")
            return None
        except Exception as exc:
            logger.exception(f"Unexpected error fetching product by item_code: {normalized_item_code}, Error: {exc}")
            return None

def get_live_product_details_by_id(composite_id: str) -> Optional[Dict[str, Any]]:
    if not composite_id:
        logger.error("get_live_product_details_by_id: Missing composite_id argument.")
        return None
    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for get_live_product_details_by_id.")
            return None
        try:
            product_entry = session.query(Product).filter_by(id=composite_id).first()
            if not product_entry:
                logger.info(f"No product entry found with composite_id: {composite_id}")
                return None
            return product_entry.to_dict()
        except SQLAlchemyError as db_exc:
            logger.exception(f"DB error fetching product by composite_id: {composite_id}, Error: {db_exc}")
            return None
        except Exception as exc:
            logger.exception(f"Unexpected error fetching product by composite_id: {composite_id}, Error: {exc}")
            return None