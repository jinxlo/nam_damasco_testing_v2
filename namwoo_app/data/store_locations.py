STORES_LIST = [
    {"city": "Caracas", "whsName": "Almacen Principal SABANA GRANDE"},
    {"city": "Caracas", "whsName": "Almacen Principal CCCT"},
    {"city": "Caracas", "whsName": "Damasco La Trinidad"},
    {"city": "Valencia", "whsName": "Tienda VALENCIA CENTRO"},
    {"city": "Valencia", "whsName": "Almacen SAN DIEGO"},
    {"city": "Maracay", "whsName": "Tienda MARACAY LAS DELICIAS"},
    {"city": "Cagua", "whsName": "Tienda CAGUA"},
    {"city": "Barquisimeto", "whsName": "Tienda BARQUISIMETO CENTRO"},
    {"city": "Maracaibo", "whsName": "Tienda MARACAIBO BELLA VISTA"},
    {"city": "Guatire", "whsName": "Tienda GUATIRE BUENAVENTURA"},
    {"city": "Lechería", "whsName": "Tienda LECHERIA C.C. Caribbean"},
    {"city": "Maturín", "whsName": "Tienda MATURIN CENTRO"},
    {"city": "Puerto Ordaz", "whsName": "Tienda PUERTO ORDAZ ALTA VISTA"},
    {"city": "Trujillo", "whsName": "Tienda TRUJILLO CENTRO"},
    {"city": "San Cristóbal", "whsName": "Tienda SAN CRISTOBAL BARRIO OBRERO"},
    {"city": "San Felipe", "whsName": "Tienda SAN FELIPE CENTRO"},
    {"city": "Puerto La Cruz", "whsName": "Tienda PUERTO LA CRUZ PLAZA MAYOR"},
    {"city": "La Guaira", "whsName": "Tienda LA GUAIRA TERMINAL"},
]

CITY_TO_WAREHOUSES = {}
for store in STORES_LIST:
    CITY_TO_WAREHOUSES.setdefault(store["city"], []).append(store["whsName"])

VALID_CITIES = sorted(CITY_TO_WAREHOUSES.keys())
