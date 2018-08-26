class Box:
    def __init__(self, class_id, score, x_min, y_min, x_max, y_max):
        self.class_id = class_id
        self.score = score
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def get_rect(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    def to_string(self):
        return ' '.join([str(a) for a in [self.class_id, self.score, self.x_min, self.y_min, self.x_max, self.y_max]])


def parse_prediction_line(line, w=1, h=1, to_ints=False):
    to_int = int if to_ints else lambda a: a
    predictions = []
    line = line.split()
    for i in range(0, len(line), 6):
        predictions += [Box(
            class_id=line[i],
            score=float(line[i+1]),
            x_min=to_int(float(line[i + 2]) * w),
            y_min=to_int(float(line[i + 3]) * h),
            x_max=to_int(float(line[i + 4]) * w),
            y_max=to_int(float(line[i + 5]) * h),
        )]
    return predictions
