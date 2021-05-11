import cv2

face_classifier = cv2.CascadeClassifier(
    "cascade/haarcascade_frontalface_default.xml")
eyes_classifier = cv2.CascadeClassifier("cascade/frontalEyes35x16.xml")

glasses = cv2.imread("img/glasses.png", cv2.IMREAD_UNCHANGED)
cont = 0

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized

def filtro_linear(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_return = face_classifier.detectMultiScale(img_gray, 1.2, 5)

    for (x, y, w, h) in faces_return:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_color = img[y:y+h, x:x+w]
        height, width, _ = roi_color.shape
        i, j = (16, 16)
        temp = cv2.resize(roi_color, (i, j), interpolation=cv2.INTER_LINEAR)
        pixelate = cv2.resize(temp, (width, height),
                              interpolation=cv2.INTER_NEAREST)
        img[y:y+h, x:x+w] = pixelate

    return img

def filtro_sobreposicao(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    faces_return = face_classifier.detectMultiScale(img_gray, 1.2, 4)

    for (x, y, w, h) in faces_return:

        roi_gray = img_gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eyes_classifier.detectMultiScale(
            roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:

            roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]
            glasses2 = image_resize(glasses.copy(), width=ew)

            gw, gh, gc = glasses2.shape
            for i in range(0, gw):
                for j in range(0, gh):
                    if glasses2[i, j][3] != 0:  # alpha 0
                        roi_color[ey + i, ex + j] = glasses2[i, j]

    return img

def mouse_click(event, x, y, flags, param):
    global cont
    if event == cv2.EVENT_LBUTTONDOWN:
        cont += 1
        while(True):
            rval, img = vc.read()
            if cont % 2 == 0:
                frame = filtro_sobreposicao(img)
                cv2.imshow('NAC02', frame)
                cv2.setMouseCallback('NAC02', mouse_click)
                cv2.waitKey(30)

            else:
                frame = filtro_linear(img)
                cv2.imshow('NAC02', frame)
                cv2.setMouseCallback('NAC02', mouse_click)
                cv2.waitKey(30)

    elif event == cv2.EVENT_RBUTTONDOWN:
        while(True):
            rval, img = vc.read()
            cv2.imshow('NAC02', img)
            cv2.setMouseCallback('NAC02', mouse_click)
            cv2.waitKey(30)

vc = cv2.VideoCapture(0)

while(vc.isOpened()):
    rval, img = vc.read()

    cv2.imshow('NAC02', img)
    cv2.setMouseCallback('NAC02', mouse_click)
    cv2.waitKey(30)

cv2.destroyAllWindows()