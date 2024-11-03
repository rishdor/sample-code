def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2
    )
    return cv2.resize(thresh, (300, 300), interpolation=cv2.INTER_AREA)

def find_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)[0]

def transform_perspective(img, contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    corners = sorted(approx.ravel().reshape(-1, 2), key=lambda x: x[1])
    ordered_corners = np.float32(
        sorted(corners[:2], key=lambda x: x[0]) + sorted(corners[2:], key=lambda x: x[0], reverse=True)
    )
    input_coords = np.float32(ordered_corners)
    output_coords = np.float32([[0, 0], [299, 0], [299, 299], [0, 299]])
    matrix = cv2.getPerspectiveTransform(input_coords, output_coords)
    return cv2.resize(cv2.warpPerspective(img, matrix, (300, 300)), (270, 270), interpolation=cv2.INTER_AREA)

def predict_number(cell):
    cell = cv2.resize(cell, (28, 28))
    cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
    cell = cell.reshape(1, 28, 28, 3)
    return np.argmax(model.predict(cell)[0])
