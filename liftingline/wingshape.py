import numpy as np


class WingShape:
    def __init__(self, airfoil_spans, airfoil_chords, airfoil_alphas, ail_spans = None):
        self.span = airfoil_spans.max()*2
        if ail_spans is not None:
            self.nr_of_controls = ail_spans.size -1
            self.ail_spans = ail_spans
        else:
            self.nr_of_controls = 0

        self.chords = airfoil_chords
        self.spans = airfoil_spans
        self.alphas = airfoil_alphas

    def chord(self, y):
        y_abs = np.abs(y)
        chord_lst = np.zeros(y.size)
        for i in range(self.chords.size-1):
            new_chords = (self.chords[i] - self.chords[i+1]) / (self.spans[i] - self.spans[i+1]) * (y_abs - self.spans[i]) + self.chords[i]
            in_range = (y_abs >= self.spans[i]) & (y_abs <= self.spans[i+1])
            chord_lst = np.where(in_range, new_chords, chord_lst)
        return chord_lst

    def alpha_geo(self, y):
        y_abs = np.abs(y)
        alpha_lst = np.zeros(y.size)
        for i in range(self.chords.size-1):
            new_alphas = (self.alphas[i] - self.alphas[i+1]) / (self.spans[i] - self.spans[i+1]) * (y_abs - self.spans[i]) + self.alphas[i]
            in_range = (y_abs >= self.spans[i]) & (y_abs <= self.spans[i+1])
            alpha_lst = np.where(in_range, new_alphas, alpha_lst)
        return alpha_lst

    def alpha_control(self, y, cont_nr):
        y_abs = np.abs(y)
        alpha_control_lst = np.zeros(y.size)
        in_range = (y >= -self.ail_spans[cont_nr+1]) & (y <= -self.ail_spans[cont_nr])
        alpha_control_lst = np.where(in_range, 1., alpha_control_lst)
        in_range = (y >= self.ail_spans[cont_nr]) & (y <= self.ail_spans[cont_nr+1])
        alpha_control_lst = np.where(in_range, -1., alpha_control_lst)
        return alpha_control_lst
