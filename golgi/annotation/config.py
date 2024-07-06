import time


class Configs:

    # IMAGE ANNOTATION CONFIGS

    current_time = time.localtime()
    
    # info
    year = current_time.tm_year
    version = "1.0"
    description = "not found"
    contributor = "not found"
    url = "not found"
    date_created = f"{current_time.tm_mday}-{current_time.tm_mon}-{current_time.tm_year}"

    default_info = {
        "year": year,
        "version": version,
        "description": description,
        "contributor": contributor,
        "url": url,
        "date_created": date_created
    }

    # license
    id = "not found"
    name = "not found"

    default_license = {
        "id": id,
        "name": name,
        "url": url
    }

    # annotation
    image_id = id
    category_id = "not found"
    segmentation = None
    area = None
    bbox = None
    iscrowd = 0

    default_annotation = {
        "id": id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "area": area,
        "bbox": bbox,
        "iscrowd": iscrowd
    }
    
    # image
    width = 0
    height = 0
    file_name = "not found"

    default_image = {
        "id": id,
        "width": width,
        "height": height,
        "file_name": file_name
    }

    default_json = {
        "info": default_info,
        "images": [default_image],
        "annotations": [default_annotation],
        "licenses": [default_license]
    }

    def get_default_annotation():
        return Configs.default_annotation.copy()

    def get_default_image():
        return Configs.default_image.copy()

    def get_default_license():
        return Configs.default_license.copy()

    def get_default_info():
        return Configs.default_info.copy()

    def get_default_json():
        return Configs.default_json.copy()

    # OTHER CONFIGS

    default_roboflow_batch = "not found"
