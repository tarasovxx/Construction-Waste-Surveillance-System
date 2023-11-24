import cv2 as cv

backSubMOG = cv.createBackgroundSubtractorMOG2()
backSubKNN = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture('test1.mp4')
if not capture.isOpened:
    print('Unable to open: ')
    exit(0)

while True:

    ret, frame = capture.read()
    if frame is None:
        break
    # print("********************************")

    fgMaskMOG = backSubMOG.apply(frame)
    fgMaskKNN = backSubKNN.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    # cv.resizeWindow('Frame', 600, 500)
    # cv.resizeWindow('FG MaskMOG', 600, 500)
    # cv.resizeWindow('FG MaskKNN', 600, 500)

    cv.imshow('Frame', frame)
    cv.imshow('FG MaskMOG', fgMaskMOG)
    cv.imshow('FG MaskKNN', fgMaskKNN)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
