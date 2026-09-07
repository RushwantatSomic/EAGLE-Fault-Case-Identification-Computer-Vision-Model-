import xml.etree.ElementTree as ET
from collections import Counter

XML = "Annotations/cvat_export.xml"

root = ET.parse(XML).getroot()

print("=" * 80)
print("EAGLE-MARS CVAT STRUCTURE INSPECTION")
print("=" * 80)

print("\nROOT:")
print(root.tag)

print("\nTOP-LEVEL ELEMENTS:")
for child in root:
    print(
        f"  {child.tag:20} "
        f"attributes={child.attrib}"
    )

print("\nTASK INFORMATION:")
print("-" * 80)

tasks = root.findall(".//task")

for task in tasks:
    print(
        f"ID={task.findtext('id')} | "
        f"NAME={task.findtext('name')} | "
        f"SOURCE={task.findtext('.//source')}"
    )

print("\nANNOTATION ELEMENT COUNTS:")
print("-" * 80)

for tag in [
    "image",
    "box",
    "polygon",
    "polyline",
    "points",
    "tag",
    "track",
    "skeleton",
]:
    elements = root.findall(f".//{tag}")
    print(f"{tag:15}: {len(elements)}")

print("\nFIRST IMAGE ELEMENT:")
print("-" * 80)

images = root.findall(".//image")

if images:
    print(
        ET.tostring(
            images[0],
            encoding="unicode"
        )[:5000]
    )
else:
    print("NO <image> ELEMENTS FOUND")

print("\nFIRST TRACK:")
print("-" * 80)

tracks = root.findall(".//track")

if tracks:
    print(
        ET.tostring(
            tracks[0],
            encoding="unicode"
        )[:5000]
    )
else:
    print("NO <track> ELEMENTS FOUND")

print("\nDONE")
