

# COCO_CLASSES_LIST = [
#     'thang',
#     'camtrai',
#     'camphai',
#     'trai',
#     'phai',
# ]

COCO_CLASSES_LIST = [
    'thang',
    'trai',
    'phai',
    'camtrai',
    'camphai',
    'None'
]


def get_cls_dict(category_num):
    if category_num == 5:
        return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
