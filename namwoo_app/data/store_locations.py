STORES_LIST = [
    {"city": "Caracas", "whsName": "ALM_CARACAS"},
    {"city": "Valencia", "whsName": "ALM_VALENCIA"},
    {"city": "Maracay", "whsName": "ALM_MARACAY"},
    {"city": "Cagua", "whsName": "ALM_CAGUA"},
    {"city": "Barquisimeto", "whsName": "ALM_BARQUISIMETO"},
    {"city": "Maracaibo", "whsName": "ALM_MARACAIBO"},
    {"city": "Guatire", "whsName": "ALM_GUATIRE"},
    {"city": "Lechería", "whsName": "ALM_LECHERIA"},
    {"city": "Maturín", "whsName": "ALM_MATURIN"},
    {"city": "Puerto Ordaz", "whsName": "ALM_PUERTO_ORDAZ"},
    {"city": "Trujillo", "whsName": "ALM_TRUJILLO"},
    {"city": "San Cristóbal", "whsName": "ALM_SAN_CRISTOBAL"},
    {"city": "San Felipe", "whsName": "ALM_SAN_FELIPE"},
    {"city": "Puerto La Cruz", "whsName": "ALM_PUERTO_LA_CRUZ"},
    {"city": "La Guaira", "whsName": "ALM_LA_GUAIRA"},
]

CITY_TO_WAREHOUSES = {}
for store in STORES_LIST:
    CITY_TO_WAREHOUSES.setdefault(store["city"], []).append(store["whsName"])

VALID_CITIES = sorted(CITY_TO_WAREHOUSES.keys())
