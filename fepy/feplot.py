import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

import logging
import numpy as np


class FECanvas:
    """Exceptions are documented in the same way as classes.

      The __init__ method may be documented in either the class level
      docstring, or as a docstring on the __init__ method itself.

      Either form is acceptable, but the two should not be mixed. Choose one
      convention to document the __init__ method and be consistent with it.

      Note:
          Do not include the `self` parameter in the ``Args`` section.

      Args:
          msg (str): Human readable string describing the exception.
          code (:obj:`int`, optional): Error code.

      Attributes:
          msg (str): Human readable string describing the exception.
          code (int): Exception error code.

      """

    __makers = ["o", "s", "+", "<", ">", ".", ",", "v", "1", "2", "3", "4"]

    __shuffle_colors = [
        "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#6A3A4C", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80", "#92896B",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#FFDBE5",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
        "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
        "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
        "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
        "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4",
        "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
        "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
        "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
        "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",
        "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
        "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
        "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
        "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
        "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
        "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
        "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
        "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",
        "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
        "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"
    ]

    def __init__(self, figure_name=None, shape=(1, 1), dpi=100):
        self.canvasMap = {}
        self.figure = plt.figure(dpi=dpi)
        if figure_name:
            self.figure.canvas.set_window_title(figure_name)
        self.shape = shape
        self.lines = []
        self.neuron_lines = dict()

        self.sub_canvas_exist = dict()

    @staticmethod
    def ax_draw_line(ax, k, b, line=None):
        """
        
        :param ax: 
        :param k: 
        :param b: 
        :param line: 
        :return: 
        """

        if line:
            x_range = line.get_xdata()
            y_range = k * x_range + b
            line.set_ydata(y_range)
            return line
        else:
            x_min, x_max = ax.get_xlim()
            x_range = np.arange(x_min, x_max, 0.05)
            y_range = k * x_range + b
            lines_arr = ax.plot(x_range, y_range)
            print("draw", b, "line_id:", lines_arr[0])
            return lines_arr[0]

    @staticmethod
    def ax_remove_line(ax, line):
        print("removing...", line)
        ax.lines.remove(line)

    def get_axis(self, sub_canvas_id=1):
        return self.add_canvas(sub_canvas_id)

    def draw_line_2d(self, k, b, color="b", sub_canvas_id=1):
        """
        
        :param k: 
        :param b: 
        :param color: 
        :param sub_canvas_id: 
        :return: 
        """

        ax = self.add_canvas(sub_canvas_id)
        line = self.ax_draw_line(ax, k, b)
        line.set_color(color)

        return line

    def remove_line_2d(self, line, sub_canvas_id=1):
        """
        
        :param line: 
        :param sub_canvas_id: 
        :return: 
        """
        ax = self.add_canvas(sub_canvas_id)
        self.ax_remove_line(ax, line)

    def add_line_2d(self, x_anchor, y_anchor, color="blue", lw=1, sub_canvas_id=1):
        """
        
        :param x_anchor: 
        :param y_anchor: 
        :param color: 
        :param lw: 
        :param sub_canvas_id: 
        :return: 
        """
        ax = self.add_canvas(sub_canvas_id)
        ax.add_line(lines.Line2D(x_anchor, y_anchor, color=color, lw=lw))

    def add_patch_rectangle(self, anchor, width, height, sub_canvas_id=1):
        """
        
        :param anchor: 
        :param width: 
        :param height: 
        :param sub_canvas_id: 
        :return: 
        """
        ax = self.add_canvas(sub_canvas_id)
        ax.add_patch(
            patches.Rectangle(
                anchor,
                width,
                height,
            )
        )

    def set_axis_lim(self, x_range, y_range, sub_canvas_id=1):
        ax = self.add_canvas(sub_canvas_id)
        if x_range:
            ax.set_xlim(x_range)
        if y_range:
            ax.set_ylim(y_range)

    def clean_canvas(self, sub_canvas_id=1):
        """
        
        :param sub_canvas_id: 
        :return: 
        """
        ax = self.add_canvas(sub_canvas_id)
        ax.clear()

    def add_canvas(self, sub_canvas_id):
        """
        
        :param sub_canvas_id: 
        :return: 
        """

        if sub_canvas_id not in self.sub_canvas_exist:
            self.sub_canvas_exist[sub_canvas_id] = self.figure.add_subplot(self.shape[0], self.shape[1], sub_canvas_id)
        # else:
        #     pass
        #     print("sub_canvas_id exists!! By default, the old canvas will not be removed")

        return self.sub_canvas_exist[sub_canvas_id]

    @staticmethod
    def set_axis_fe_style(ax):
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('center')
        return ax

    def draw_line_chart_2d(self, x_data, y_data, sub_canvas_id=1, color="#03A9F4", line_style="dashed", fe_style=True):

        ax = self.add_canvas(sub_canvas_id)
        ax.plot(x_data, y_data, color=color, linestyle=line_style)

        if fe_style:
            ax = FECanvas.set_axis_fe_style(ax)
        return ax

    def draw_classification_data_point_2d(self, data, class_index, sub_canvas_id=1, marker_size=30):
        """
        
        :param data: 
        :param class_index: 
        :param sub_canvas_id: 
        :param marker_size: 
        :return: 
        """

        ax = self.add_canvas(sub_canvas_id)

        class_style_dict = dict()
        style_counter = 0

        index_arr = [0, 1, 2]
        index_arr.pop(class_index)

        logging.debug(data)

        for row in data:
            key = row[class_index]
            if not class_style_dict.get(key):
                try:
                    class_style_dict[key] = style_counter
                    style_counter += 1
                except IndexError:
                    print("The length of color-style-array is smaller than the number of categories!!")
                    exit()

            class_style_index = class_style_dict[key]

            point_color = self.__shuffle_colors[class_style_index]
            point_style = self.__makers[class_style_index]

            x = row[index_arr[0]]
            y = row[index_arr[1]]
            ax.scatter(x, y, c=point_color, marker=point_style, s=marker_size)

        return ax

    def draw_neuron_lines(self, neuron_matrix, sub_canvas_id=1):
        """
        
        :param neuron_matrix: 
        :param sub_canvas_id: 
        :return: 
        """

        weight_matrix = neuron_matrix.weight
        bias_matrix = neuron_matrix.bias
        number_of_lines = weight_matrix.shape[0]

        ax = self.add_canvas(sub_canvas_id)
        for i in range(number_of_lines):
            k = -1 * weight_matrix[i, 0] / weight_matrix[i, 1]
            b = -1 * bias_matrix[i, 0] / weight_matrix[i, 1]

            if sub_canvas_id not in self.neuron_lines:
                self.neuron_lines[sub_canvas_id] = dict()

            if i in self.neuron_lines[sub_canvas_id]:
                line = self.neuron_lines[sub_canvas_id][i]
                self.ax_draw_line(ax, k, b, line)
            else:
                line = self.draw_line_2d(k, b, sub_canvas_id=sub_canvas_id)
                line.set_color(self.__shuffle_colors[i+2])
                self.neuron_lines[sub_canvas_id][i] = line

    def set_title(self, title, sub_canvas_id=1):
        ax = self.add_canvas(sub_canvas_id)
        ax.set_title(title)

    def set_x_label(self, label, sub_canvas_id=1):
        ax = self.add_canvas(sub_canvas_id)
        ax.set_xlabel(label)

    def set_y_label(self, label, sub_canvas_id=1):
        ax = self.add_canvas(sub_canvas_id)
        ax.set_ylabel(label)

    def set_label(self, x_label, y_label, sub_canvas_id=1):
        self.set_x_label(x_label, sub_canvas_id)
        self.set_y_label(y_label, sub_canvas_id)

    def show(self, pause_interval=10):
        self.figure.show()
        plt.pause(pause_interval)

    def froze(self):
        self.figure.show()
        plt.show()

    def save(self, filename, sub_canvas_id=1):
        self.figure.savefig(filename)
        plt.close(self.figure)
