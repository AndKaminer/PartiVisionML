import datetime

def parse_model_date(name):
    if not name.endswith(".pt"):
        raise Exception("Invalid filetype")

    name = name[:-3]

    split = name.split("-")
    if len(split) < 3:
        raise Exception("Date not found")

    day, month, year = int(split[-3]), int(split[-2]), int(split[-1])

    return datetime.date(year, month, day)


def hasdate(name):
    if not name.endswith(".pt"):
        raise Exception("Invalid filetype")

    name = name[:-3]

    split = name.split("-")

    if len(split) < 3:
        return False

    day, month, year = split[-3], split[-2], split[-1]

    return day.isdigit() and month.isdigit() and year.isdigit()
