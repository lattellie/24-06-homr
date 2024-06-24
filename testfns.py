from homr.bounding_boxes import BoundingEllipse
from homr.note_detection import NoteheadWithStem
import cv2
import numpy as np

def visualizeEllipses(noteheads,image, type, color=(0,0,255), thickness = -1, dirPath=''):
    img = image.copy()
    for i in range(len(noteheads)):
        n = noteheads[i]
        img = cv2.ellipse(img, tuple(int(i) for i in n.center), tuple(int(i/2) for i in n.size), int(n.angle), 0, 360, color=color, thickness=thickness)
    showResizedImg(img)
    cv2.imwrite(dirPath+type+'.jpg', img)
    print(f'image wrote to {dirPath+type}.jpg')

def visualizeBoundingBoxes(staff_fragment, image, type, color=(0,0,255), thickness = 2, dirPath = ''):
    print('visualizing bounding boxes')
    img = image.copy()
    for i in range(len(staff_fragment)):
        sf = staff_fragment[i]
        img = cv2.rectangle(img, tuple(int(i) for i in sf.top_left), tuple(int(i) for i in sf.bottom_right), color, thickness, cv2.LINE_AA)
    showResizedImg(img)
    cv2.imwrite(dirPath+type+'.jpg', img)
    print(f'image wrote to {dirPath+type}.jpg')

def visualizeRotatedRect(staff_fragment, image, type, color=(0,0,255), thickness = 2, dirPath = ''):
    print('visualizing rotated rect')
    img = image.copy()
    for i in range(len(staff_fragment)):
        sf = staff_fragment[i]
        img = cv2.line(img, tuple(int(i) for i in sf.top_left), tuple(int(i) for i in sf.top_right), color, thickness)
        img = cv2.line(img, tuple(int(i) for i in sf.bottom_right), tuple(int(i) for i in sf.top_right), color, thickness)
        img = cv2.line(img, tuple(int(i) for i in sf.bottom_right), tuple(int(i) for i in sf.bottom_left), color, thickness)
        img = cv2.line(img, tuple(int(i) for i in sf.bottom_left), tuple(int(i) for i in sf.top_left), color, thickness)
    showResizedImg(img)
    cv2.imwrite(dirPath+type+'.jpg', img)
    print(f'image wrote to {dirPath+type}.jpg')

def visualizeNoteHeadWithStems(noteheadWithStem, image, type, thickness = 2, dirPath = ''):
    print('visualizing notehead with stems')
    img = image.copy()
    allcolors = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(255,0,255),(120,120,0),(0,120,120),(120,0,120)]
    for i in range(len(noteheadWithStem)):
        nhws:NoteheadWithStem = noteheadWithStem[i]
        nh = nhws.notehead
        stm = nhws.stem
        color = allcolors[i%(len(allcolors))]
        img = cv2.ellipse(img, tuple(int(i) for i in nh.center), tuple(int(i/2) for i in nh.size), int(nh.angle), 0, 360, color=color, thickness=thickness)
        if stm is None:
            print(f'no stem for {i}')
            continue
        img = cv2.line(img, tuple(int(i) for i in stm.top_left), tuple(int(i) for i in stm.top_right), color, thickness)
        img = cv2.line(img, tuple(int(i) for i in stm.bottom_right), tuple(int(i) for i in stm.top_right), color, thickness)
        img = cv2.line(img, tuple(int(i) for i in stm.bottom_right), tuple(int(i) for i in stm.bottom_left), color, thickness)
        img = cv2.line(img, tuple(int(i) for i in stm.bottom_left), tuple(int(i) for i in stm.top_left), color, thickness)

    showResizedImg(img)
    cv2.imwrite(dirPath+type+'.jpg', img)
    print(f'image wrote to {dirPath+type}.jpg')

def visualizeStaffs(staffs, image, type, thickness=1, color = (0,0,255),dirPath = ''):
    print('visualizing staffs')
    img = image.copy()
    for i in range(len(staffs)):
        sf = staffs[i]
        img = cv2.rectangle(img, (int(sf.min_x), int(sf.min_y)),(int(sf.max_x), int(sf.max_y)), color, thickness, cv2.LINE_AA)
    showResizedImg(img)
    cv2.imwrite(dirPath+type+'.jpg', img)
    print(f'image wrote to {dirPath+type}.jpg')    

def visualizeBarlines(bar_lines_found, image, type, thickness=2, color=(0,0,255), dirPath = ''):
    print('visualizing Barlines')
    img = image.copy()
    for i in range(len(bar_lines_found)):
        bl = bar_lines_found[i]
        box = bl.box
        img = cv2.rectangle(img, tuple(int(i) for i in box.top_left), tuple(int(i) for i in box.bottom_right), color, thickness, cv2.LINE_AA)
    showResizedImg(img)
    cv2.imwrite(dirPath+type+'.jpg', img)
    print(f'image wrote to {dirPath+type}.jpg')    
    
def visualizeNotes(notes, image, type, thickness=1, color=(0,0,255), dirPath = ''):
    print('visualizing Notes')
    img = image.copy()
    for i in range(len(notes)):
        n = notes[i]
        box = n.box
        if n.notehead_type.best.value == 1:
            color = (255,255,0) # hollow
        else:
            color = (0,0,255)
        img = cv2.rectangle(img, tuple(int(i) for i in box.top_left), tuple(int(i) for i in box.bottom_right), color, thickness, cv2.LINE_AA)
        disptext = str(n.position)+("." if n.has_dot else "")
        img = cv2.putText(img, disptext, tuple(int(i) for i in box.bottom_right),cv2.FONT_HERSHEY_SIMPLEX,  0.5, color, thickness, cv2.LINE_AA)

    showResizedImg(img)
    cv2.imwrite(dirPath+type+'.jpg', img)
    print(f'image wrote to {dirPath+type}.jpg')    

def visualizeBinary(nparray, image, type, imageAlpha=1, color=(0,0,255), dirPath = ''):
    print('visualizing binary array')
    img = image.copy()
    residual = np.subtract(255,img)
    img255 = img.copy()
    img255.fill(255)
    img = (img255- residual*imageAlpha)
    img[nparray>0] = color
    img = img.astype(np.uint8)
    print(img.shape)
    showResizedImg(img)
    cv2.imwrite(dirPath+type+'.jpg', img)
    print(f'image wrote to {dirPath+type}.jpg')    
    
def showResizedImg(img, resize_ratio=0.3):
    resized_image = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation = cv2.INTER_AREA)
    cv2.imshow("resized image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
