"""schema.py — OData $metadata fetch + prompt block for the Northwind example."""

import re

import requests

NORTHWIND_URL = "https://services.odata.org/V4/Northwind/Northwind.svc"

_schema: dict = {}


def _entity_type_for(entity_set: str) -> str:
    """Map an entity set name (plural) to its entity type name (singular).
    e.g. 'Orders' -> 'Order', 'Products' -> 'Product'.
    Falls back to the set name itself if no singular form is known.
    """
    if entity_set.endswith("s") and not entity_set.endswith("ss"):
        return entity_set[:-1]
    return entity_set


def fetch_schema(entities: list = None) -> dict:
    """Fetch the real field list for each entity from the OData $metadata.

    Returns a dict mapping entity set name -> list of field names.
    Caches the result in _schema. On failure, returns the current _schema
    (possibly empty) so the caller can fall back to a hardcoded schema.
    """
    if entities is None:
        entities = ["Orders", "Products", "Customers", "Employees", "Categories", "Order_Details"]
    try:
        resp = requests.get(f"{NORTHWIND_URL}/$metadata", timeout=30)
        resp.raise_for_status()
        text = resp.text
        for entity_set in entities:
            entity_type = _entity_type_for(entity_set)
            pattern = r'<EntityType Name="' + entity_type + r'">.*?</EntityType>'
            m = re.search(pattern, text, re.DOTALL)
            if m:
                fields = re.findall(r'<Property Name="([^"]+)"', m.group(0))
                if fields:
                    _schema[entity_set] = fields
    except requests.RequestException:
        pass  # fall back to whatever _schema already has (or empty)
    return _schema


FIELD_DESCRIPTIONS = {
    "Orders": {
        "OrderID": "Unique order identifier",
        "CustomerID": "ID of the customer who placed the order (links to Customers.CustomerID)",
        "EmployeeID": "ID of the employee who took/handled the order (links to Employees.EmployeeID)",
        "OrderDate": "Date the order was placed",
        "RequiredDate": "Date the order is required to be shipped by (delivery deadline)",
        "ShippedDate": "Date the order was actually shipped (may be null if not yet shipped)",
        "ShipVia": "ID of the shipper/carrier used for delivery",
        "Freight": "Shipping cost charged to the customer (in USD)",
        "ShipName": "Name of the shipping recipient / shipping partner / shipper",
        "ShipAddress": "Street address of the shipping destination",
        "ShipCity": "City of the shipping destination",
        "ShipRegion": "Region/state/province of the shipping destination",
        "ShipPostalCode": "Postal/ZIP code of the shipping destination",
        "ShipCountry": "Country of the shipping destination",
    },
    "Products": {
        "ProductID": "Unique product identifier",
        "ProductName": "Name of the product",
        "SupplierID": "ID of the supplier who provides this product",
        "CategoryID": "ID of the product category (links to Categories.CategoryID)",
        "QuantityPerUnit": "Package size/description (e.g., '10 - 500 g cans')",
        "UnitPrice": "Price per unit (in USD)",
        "UnitsInStock": "Number of units currently in stock (0 = out of stock)",
        "UnitsOnOrder": "Number of units ordered from supplier but not yet received",
        "ReorderLevel": "Stock level at which reorder alerts are triggered",
        "Discontinued": "True if the product is no longer available",
    },
    "Customers": {
        "CustomerID": "Unique customer identifier (links to Orders.CustomerID)",
        "CompanyName": "Name of the customer company",
        "ContactName": "Name of the primary contact person",
        "ContactTitle": "Job title of the contact person",
        "Address": "Street address of the customer",
        "City": "City of the customer",
        "Region": "Region/state/province of the customer",
        "PostalCode": "Postal/ZIP code of the customer",
        "Country": "Country of the customer",
        "Phone": "Phone number of the customer",
        "Fax": "Fax number of the customer",
    },
    "Employees": {
        "EmployeeID": "Unique employee identifier (links to Orders.EmployeeID)",
        "LastName": "Employee last name",
        "FirstName": "Employee first name",
        "Title": "Job title",
        "TitleOfCourtesy": "Courtesy title (Mr., Mrs., Dr., etc.)",
        "BirthDate": "Date of birth",
        "HireDate": "Date the employee was hired",
        "Address": "Home address",
        "City": "Home city",
        "Region": "Home region/state",
        "PostalCode": "Home postal code",
        "Country": "Home country",
        "HomePhone": "Home phone number",
        "Extension": "Office phone extension",
        "Photo": "Employee photo (binary, ignore)",
        "Notes": "Internal notes about the employee",
        "ReportsTo": "EmployeeID of the manager this employee reports to",
        "PhotoPath": "File path to employee photo",
    },
    "Categories": {
        "CategoryID": "Unique category identifier (links to Products.CategoryID)",
        "CategoryName": "Name of the product category",
        "Description": "Description of the category",
        "Picture": "Category image (binary, ignore)",
    },
    "Order_Details": {
        "OrderID": "Order this line item belongs to (links to Orders.OrderID)",
        "ProductID": "Product in this line item (links to Products.ProductID)",
        "UnitPrice": "Price per unit at time of order",
        "Quantity": "Number of units ordered",
        "Discount": "Discount applied to this line item (0 to 1)",
    },
}


def schema_prompt_block() -> str:
    """Build a prompt-ready string of the schema with field descriptions."""
    if not _schema:
        return "(schema unavailable — call odata_metadata to discover fields)"
    lines = []
    for entity_set, fields in _schema.items():
        descs = FIELD_DESCRIPTIONS.get(entity_set, {})
        field_parts = []
        for f in fields:
            desc = descs.get(f, "")
            if desc:
                field_parts.append(f"{f} ({desc})")
            else:
                field_parts.append(f)
        lines.append(f"- {entity_set}: {', '.join(field_parts)}")
    return "\n".join(lines)
