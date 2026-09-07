import requests

CVAT_URL = "http://localhost:8080"

TASK_IDS = [
    2523742,
    2523747,
    2523748,
    2524033,
    2524047,
    2524049,
    2524050,
    2524051,
    2524052,
    2524053,
    2524054,
    2524056,
    2524057,
    2524059,
    2524061,
    2524062,
    2524063,
    2524064,
    2524065,
    2524068,
    2526254,
]

print("=" * 100)
print("CVAT TASK METADATA")
print("=" * 100)

for task_id in TASK_IDS:

    url = f"{CVAT_URL}/api/tasks/{task_id}"

    try:
        response = requests.get(url, timeout=10)

        print("\n" + "-" * 100)
        print("TASK:", task_id)
        print("HTTP:", response.status_code)

        if response.ok:
            data = response.json()

            for key, value in data.items():
                print(f"{key}: {value}")

        else:
            print(response.text[:1000])

    except Exception as e:
        print("ERROR:", e)

print("\n" + "=" * 100)
print("DONE")