import numpy as np
import matplotlib.pyplot as plt
import cv2

class UpdateablePlot:
    def __init__(self, num_subplots=1, height=3.2, typ='plot', title='Plots'):
        # valid types: 'scatter', 'plot'
        self.num_subplots = num_subplots
        self.height = height
        self.width = num_subplots * height * 1.2
        self.fig = plt.figure(figsize=(self.width, self.height))
        self.fig.tight_layout(rect=[0, 0, .5, 1])
        self.fig.suptitle(title)
        self.axes = []
        self.typ = typ
        self.num_subplots = num_subplots
        self.title = title
        for i in range(num_subplots):
            ax = self.fig.add_subplot(1, num_subplots, i+1)

            if typ == 'plot':
                data, = ax.plot([], [])
            elif typ == 'scatter':
                data = ax.scatter([], [])
            else:
                raise Exception("Invalid type")

            self.axes.append((ax, data))

        for i in range(num_subplots):
            self.change_subplot_dims(i)

        plt.close()

    def reset(self):
        self.fig = plt.figure(figsize=(self.width, self.height))
        self.fig.tight_layout()
        self.fig.suptitle(self.title)
        self.axes = []
        for i in range(self.num_subplots):
            ax = self.fig.add_subplot(1, self.num_subplots, i+1)

            if self.typ == 'plot':
                data, = ax.plot([], [])
            elif self.typ == 'scatter':
                data = ax.scatter([], [])
            else:
                raise Exception("Invalid type")

            self.axes.append((ax, data))

        for i in range(self.num_subplots):
            self.change_subplot_dims(i)

        plt.close()


    def update_subplot(self, subplot_idx, xval, yval, padding_ratio):
        if subplot_idx not in range(len(self.axes)):
            raise Exception("Invalid index")

        ax, data, = self.axes[subplot_idx]

        if self.typ == 'plot':
            xdata, ydata = data.get_data()
            xdata = list(xdata)
            ydata = list(ydata)
            xdata.append(xval)
            ydata.append(yval)
            data.set_data(xdata, ydata)

            max_val = max(ydata) if ydata else 0
            min_val = min(ydata) if ydata else -0
            xlim = (xdata[0], xdata[-1])

        elif self.typ == 'scatter':
            points = data.get_offsets().data
            new_data = np.array([[xval, yval]])
            points = np.append(points, new_data, axis=0)
            data.set_offsets(points)
            max_val = max(points, key=lambda b : b[1])[1]
            min_val = min(points, key=lambda b : b[1])[1]
            xlim = (points[0][0], points[-1][0])
        else:
            raise Exception("Invalid type")
        
        datarange = max_val - min_val
        max_val = max_val + (padding_ratio / 2 * datarange)
        min_val = min_val - (padding_ratio / 2 * datarange)
        ylim = (min_val, max_val)

        if ylim[0] == ylim[1]:
            ylim = (1, -1)

        if xlim[0] == xlim[1]:
            xlim = (1, -1)

        ax.set_ylim(ylim)
        ax.set_xlim(xlim)


    def set_subplot_chars(self, subplot_idx, title, xlabel, ylabel, fontsize=10):
        if subplot_idx not in range(len(self.axes)):
            raise Exception("Invalid index")

        ax, data, = self.axes[subplot_idx]

        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)

    def change_subplot_dims(self, subplot_idx, xlim=(0,10), ylim=(0,10)):
        if subplot_idx not in range(len(self.axes)):
            raise Exception("Invalid index")

        ax, data, = self.axes[subplot_idx]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def get_img(self):
        fig = self.fig
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        return img
