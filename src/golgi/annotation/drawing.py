import cv2

class Drawing:
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask.copy()
        
        self.state = {
            "drawing": False,
            "white": False,
            "show_mask": False,
            "show_contours": True
        }

        self.window_name = "Draw Contour"

    def draw(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.window_name, (1920, 1080))
        cv2.setMouseCallback(self.window_name, self._draw_circle)
        
        while True:
            self._draw_frame()
            key = cv2.waitKey(1)

            if key == 27:
                break
            elif key == ord('m'):
                self.state["show_mask"] = not self.state["show_mask"]
            elif key == ord('c'):
                self.state["white"] = not self.state["white"]
            elif key == ord('o'):
                self.state["show_contours"] = not self.state["show_contours"]


        cv2.destroyWindow(self.window_name)

        return self.mask

    def _draw_frame(self):
        if self.state["show_mask"]:
            cv2.imshow(self.window_name, self.mask)
            return

        frame = self.image.copy()
        if self.state["show_contours"]:
            contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 1) 

        cv2.imshow(self.window_name, frame)


    def _draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.state["drawing"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.state["drawing"] = False
            
        if self.state["drawing"]:
            color = (255, 255, 255) if self.state["white"] else (0, 0, 0)
            cv2.circle(self.mask, (x, y), 3, color, -1)
