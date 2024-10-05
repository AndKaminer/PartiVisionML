from ..plotting import UpdateablePlot

class DataHandler:

    def __init__(self, deltatime=1, scatter=False):

        self.time = 0
        self.deltatime = deltatime
        self.scatter = scatter

        self.data = {
            "area": [],
            "height": [],
            "perimeter": [],
            "velocity": [],
            "acceleration": [],
            "circularity": [],
            "time": []}

        self.prev_data = {
            'area' : 0,
            'perimeter' : 0,
            'height' : 0,
            'velocity' : 0,
            'acceleration' : 0,
            'circularity' : 0,
            'centerX' : None}

        self.plot = UpdateablePlot(6, 3.2, 'scatter') if scatter else UpdateablePlot(6, 3.2, 'plot')
        self.plot.set_subplot_chars(0, "Area", "Time (us)", "Area (um^2)")
        self.plot.set_subplot_chars(1, "Perimeter", "Time (us)",  "Perimeter (um)")
        self.plot.set_subplot_chars(2, "Height", "Time (us)", "Height (um)")
        self.plot.set_subplot_chars(3, "Velocity", "Time (us)", "Velocity (um/s)")
        self.plot.set_subplot_chars(4, "Acceleration", "Time (us)", "Acceleration (um/s^2)")
        self.plot.set_subplot_chars(5, "Circularity", "Time (us)", "Circularity")


    def update_data(self, area, perimeter, height, circularity, centerX=None):

        self.time += self.deltatime
        self.data['time'].append(self.time)

        if centerX and self.prev_data['centerX']:
            velocity = (centerX - self.prev_data['centerX']) / self.deltatime
        else:
            velocity = 0

        acceleration = (velocity - self.prev_data['velocity']) / self.deltatime

        self.data['area'].append(area)
        self.prev_data['area'] = area
        self.data['height'].append(height)
        self.prev_data['height'] = height
        self.data['perimeter'].append(perimeter)
        self.prev_data['perimeter'] = perimeter
        self.data['velocity'].append(velocity)
        self.prev_data['velocity'] = velocity
        self.data['acceleration'].append(acceleration)
        self.prev_data['acceleration'] = acceleration
        self.data['circularity'].append(circularity)
        self.prev_data['circularity'] = circularity
        self.prev_data['centerX'] = centerX

        self.__update_plot()

    
    def get_plot_img(self):
        return self.plot.get_img()


    def __update_plot(self):
        self.plot.update_subplot(0, self.time, self.data['area'][-1], .1)
        self.plot.update_subplot(1, self.time, self.data['perimeter'][-1], .1)
        self.plot.update_subplot(2, self.time, self.data['height'][-1], .1)
        self.plot.update_subplot(3, self.time, self.data['velocity'][-1], .1)
        self.plot.update_subplot(4, self.time, self.data['acceleration'][-1], .1)
        self.plot.update_subplot(5, self.time, self.data['circularity'][-1], .1)
