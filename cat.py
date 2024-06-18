class Point:
    def __init__(self, x: int, y: int) -> None:
        self._x = x
        self._y = y

    def getX(self) -> int:
        return self._x

    def getY(self) -> int:
        return self._y

    def setX(self, x: int) -> None:
        self._x = x

    def setY(self, y: int) -> None:
        self._y = y

    def getXY(self) -> tuple:
        return (self._x, self._y)

    def setXY(self, x: int, y: int) -> None:
        self._x = x
        self._y = y

    def __str__(self) -> str:
        return f'Point ({self._x}, {self._y})'


class Ear:
    def __init__(self, p1: Point, p2: Point, p3: Point) -> None:
        self._p1 = p1
        self._p2 = p2
        self._p3 = p3

    def getP1(self) -> Point:
        return self._p1

    def getP2(self) -> Point:
        return self._p2

    def getP3(self) -> Point:
        return self._p3

    def getPoints(self) -> tuple:
        return (self._p1, self._p2, self._p3)

    def __str__(self) -> str:
        return f'Ear: ({self._p1}, {self._p2}, {self._p3})'


class Face:
    def __init__(self, image_file_path: str, coords_file_path: str, leftEye: Point, rightEye: Point, mouth: Point, leftEar: Ear, rightEar: Ear) -> None:
        self._image_file_path = image_file_path
        self._coords_file_path = coords_file_path

        self._leftEye = leftEye
        self._rightEye = rightEye
        self._mouth = mouth
        self._leftEar = leftEar
        self._rightEar = rightEar

    def getFilePath(self) -> str:
        return self._file_path

    def getLeftEye(self) -> Point:
        return self._leftEye

    def getRightEye(self) -> Point:
        return self._rightEye

    def getMouth(self) -> Point:
        return self._mouth

    def getLeftEar(self) -> Ear:
        return self._leftEar

    def getRightEar(self) -> Ear:
        return self._rightEar

    @staticmethod
    def fromFile(image_file_path: str, coords_file_path: str) -> 'Face':
        with open(coords_file_path, 'r') as f:
            coords = [int(i) for i in f.readline().split()[1:]]
            leftEye = Point(coords[0], coords[1])
            rightEye = Point(coords[2], coords[3])
            mouth = Point(coords[4], coords[5])
            leftEar = Ear(Point(coords[6], coords[7]), Point(
                coords[8], coords[9]), Point(coords[10], coords[11]))
            rightEar = Ear(Point(coords[12], coords[13]), Point(
                coords[14], coords[15]), Point(coords[16], coords[17]))

        return Face(image_file_path, coords_file_path, leftEye, rightEye, mouth, leftEar, rightEar)

    def __str__(self) -> str:
        return f'Face: {self._image_file_path} ({self._leftEye}, {self._rightEye}, {self._mouth}, {self._leftEar}, {self._rightEar})'