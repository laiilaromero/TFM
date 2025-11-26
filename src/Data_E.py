import ifcopenshell
import ifcopenshell.geom
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
import pandas as pd
from pathlib import Path

# -----------------------------------------
# Configuración de geometría
# -----------------------------------------
settings = ifcopenshell.geom.settings()
settings.set("use_python_opencascade", True)
settings.set("USE_WORLD_COORDS", True)
settings.set("DISABLE_OPENING_SUBTRACTIONS", True)

# -----------------------------------------
# Cargar IFC
# -----------------------------------------
ROOT = Path(__file__).resolve().parents[1]
Data_dir = ROOT / "Data"
ifc_path = Data_dir /"Clinic_Structural.ifc"
model = ifcopenshell.open(ifc_path)

# -----------------------------------------
# Elementos y clases a analizar
# -----------------------------------------
target_classes = ["IfcWall", "IfcSlab", "IfcBeam", "IfcColumn"]
all_elements = model.by_type("IfcElement")

rows = []

for elem in all_elements:
    # Material y espesor
    material_name = None
    layer_thickness = None
    for r in model.get_inverse(elem):
        if r.is_a("IfcRelAssociatesMaterial"):
            mat = r.RelatingMaterial
            if mat.is_a("IfcMaterialLayerSetUsage"):
                for layer in mat.ForLayerSet.MaterialLayers:
                    material_name = layer.Material.Name
                    layer_thickness = layer.LayerThickness

    # Geometría: área y volumen
    try:
        shape = ifcopenshell.geom.create_shape(settings, elem).geometry
        props = GProp_GProps()
        brepgprop_VolumeProperties(shape, props)
        vol = props.Mass()  # m3

        props = GProp_GProps()
        brepgprop_SurfaceProperties(shape, props)
        area = props.Mass()  # m2
    except:
        vol = None
        area = None

    rows.append({
        "id": elem.id(),
        "type": elem.is_a(),
        "GlobalId": getattr(elem, "GlobalId", None),
        "Name": getattr(elem, "Name", None),
        "material_name": material_name,
        "Espesor_mm": layer_thickness,
        "area_m2": area,
        "volume_m3": vol,
        "objectType": getattr(elem, "ObjectType", None),
        "is_target_class": elem.is_a() in target_classes
    })

# -----------------------------------------
# Crear DataFrame
# -----------------------------------------
df_ifc = pd.DataFrame(rows)

# Separar por elementos target si quieres
df_targets = df_ifc[df_ifc["is_target_class"]]

# -----------------------------------------
# Mostrar resultados
# -----------------------------------------
#print("=== Elementos Target ===")
#print(df_targets)

print("\n=== Todos los elementos ===")
print(df_ifc.head())
#Exporta Datframes
OUTPUT_DIR  = ROOT/"outputs"
CSV_DIR = OUTPUT_DIR / "csv"
df_ifc.to_csv(CSV_DIR/"Ifc_all_elements.csv",index=False, sep=";")
df_targets.to_csv(CSV_DIR/"ifc_target_elements.csv", index=False, sep=";")
print("\n CSV guardado")
print("- ifc_all_elements.csv (todos los elementos")
print(" - ifc_target_elements.csv (solo Walls, Slabs, Beams, Columns)")

