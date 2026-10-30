"""Entrapment labeller: the third-person video above the robot's signals, with the labels drawn on a shared time axis.

usage: python labeller/label.py <dataset folder> <sequence> --annotator A1 [--out <labels.yaml>]
       python labeller/label.py <dataset folder> <sequence> --annotator consensus --compare A1 A2

Labels are saved after every change; an existing label file is opened and continued. --compare shows two
annotators' labels (names from the walk's annotations folder, or label files) and lists where they disagree.
"""
import argparse
import copy
import sys
from pathlib import Path
import av
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QHBoxLayout, QLabel, QListWidget, QMainWindow,
                               QPushButton, QSplitter, QVBoxLayout, QWidget)
import labels
from data import BEHIND, DT, Walk, walk_dir
from labels import Entrapment, Intervention, TYPES

COLOURS = {'entrapment': (70, 130, 220), 'controller': (240, 150, 40), 'physical': (220, 60, 60),
           'candidate': (150, 150, 150), 'other': (120, 120, 120), 'unreliable': (240, 200, 120)}


def brush(name, alpha):
    return pg.mkBrush(*COLOURS[name], alpha)


class Video:
    """Frames of the third-person video by time: decoded forward, seeking only on jumps."""

    def __init__(self, path, width=960):
        self.c = av.open(str(path))
        self.s = self.c.streams.video[0]
        self.s.thread_type = 'AUTO'
        self.width = width
        self.height = round(self.s.codec_context.height * width / self.s.codec_context.width / 2) * 2
        self.cur = self.ahead = self.frames = None

    def image(self, t):
        """The frame on screen at video time t, as a QImage."""
        if self.cur is None or t < self.cur.time or t > self.cur.time + 2:
            self.c.seek(int(max(t, 0) / self.s.time_base), stream=self.s, backward=True)
            self.frames = self.c.decode(self.s)
            self.cur = self.ahead = None
        while self.ahead is None or self.ahead.time <= t:
            if self.ahead is not None:
                self.cur = self.ahead
            self.ahead = next(self.frames, None)
            if self.ahead is None:
                break
        if self.cur is None:
            return None
        a = self.cur.to_ndarray(format='rgb24', width=self.width, height=self.height)
        return QImage(a.data, self.width, self.height, 3 * self.width, QImage.Format_RGB888).copy()


class Sticks(QWidget):
    """The controller sticks at the cursor, with the last second as a trail."""

    def __init__(self):
        super().__init__()
        self.setFixedWidth(170)
        self.left = self.right = np.zeros((0, 2))

    def show_at(self, stick, i):
        s = stick[max(0, i - int(1 / DT)):i + 1]
        self.left = s[:, [0, 1]]
        self.right = np.c_[s[:, 2], np.zeros(len(s))]
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        size = min(self.width() - 30, self.height() / 2 - 40)
        for k, (title, trail) in enumerate((('left stick: forward, sideways', self.left),
                                            ('right stick: turn', self.right))):
            top = 10 + k * self.height() / 2
            p.setPen(Qt.black)
            p.drawText(QRectF(0, top, self.width(), 16), Qt.AlignHCenter, title)
            box = QRectF((self.width() - size) / 2, top + 20, size, size)
            p.setPen(QPen(Qt.gray, 1))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(box)
            c, r = box.center(), size / 2

            def at(v):
                x, y = np.clip(np.nan_to_num(v), -1, 1)
                return QPointF(c.x() + x * r, c.y() - y * r)

            for j in range(1, len(trail)):
                p.setPen(QPen(pg.mkColor(70, 130, 220, int(40 + 200 * j / len(trail))), 2))
                p.drawLine(at(trail[j - 1]), at(trail[j]))
            if len(trail) and np.isfinite(trail[-1]).all():
                p.setBrush(pg.mkColor(70, 130, 220))
                p.drawEllipse(at(trail[-1]), 6, 6)
        p.end()


class DragLane(pg.ViewBox):
    """The entrapment lane: dragging along an empty part draws a new entrapment."""

    def __init__(self, done):
        super().__init__()
        self.done = done
        self.preview = None

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() != Qt.LeftButton:
            return super().mouseDragEvent(ev, axis)
        ev.accept()
        a, b = sorted(self.mapSceneToView(q).x() for q in (ev.buttonDownScenePos(), ev.scenePos()))
        if self.preview is None:
            self.preview = pg.LinearRegionItem((a, b), movable=False, brush=brush('entrapment', 60))
            self.addItem(self.preview)
        self.preview.setRegion((a, b))
        if ev.isFinish():
            self.removeItem(self.preview)
            self.preview = None
            if b - a > 0.2:
                self.done(a, b)


class Window(QMainWindow):
    def __init__(self, walk, path, annotator, compare):
        super().__init__()
        self.walk, self.path, self.annotator, self.compare = walk, path, annotator, compare
        self.entrapments = labels.load(path)
        self.history = []
        self.selected = self.shown = None
        self.time = 0.0
        self.duration = walk.t[-1]
        self.video = Video(walk.video) if walk.video_offset is not None else None
        self.step = 1 / walk.fps if self.video else DT
        self.setWindowTitle(f'{walk.seq} - {annotator} - {path}')

        self.frame = QLabel('no video' if self.video is None else '')
        self.frame.setAlignment(Qt.AlignCenter)
        self.frame.setMinimumSize(480, 270)
        self.sticks = Sticks()
        self.status = QLabel()
        self.lanes = pg.GraphicsLayoutWidget()
        self.build_lanes()
        top = QWidget()
        row = QHBoxLayout(top)
        row.addWidget(self.frame, 1)
        row.addWidget(self.sticks)
        left = QSplitter(Qt.Vertical)
        left.addWidget(top)
        left.addWidget(self.lanes)
        left.setSizes([330, 620])

        body = QHBoxLayout()
        body.addWidget(left, 1)
        body.addWidget(self.build_panel())
        main = QVBoxLayout()
        main.addLayout(body)
        main.addWidget(self.status)
        root = QWidget()
        root.setLayout(main)
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        for key, f in (('Space', self.play), ('Left', lambda: self.set_time(self.time - self.step)),
                       ('Right', lambda: self.set_time(self.time + self.step)),
                       ('Shift+Left', lambda: self.set_time(self.time - 1)),
                       ('Shift+Right', lambda: self.set_time(self.time + 1)), ('Ctrl+Z', self.undo)):
            QShortcut(QKeySequence(key), self, f)
        self.refresh()
        self.set_time(0.0)

    def build_panel(self):
        actions = [('New entrapment here', 'N', self.new_entrapment),
                   ('From candidate', 'A', self.from_candidate),
                   ('Start here', 'B', lambda: self.move_edge(0)),
                   ('End here', 'E', lambda: self.move_edge(1)),
                   ('Controller starts', 'C', lambda: self.add_intervention('controller')),
                   ('Controller ends', 'Shift+C', self.end_controller),
                   ('Physical starts', 'P', lambda: self.add_intervention('physical')),
                   ('Delete intervention', '', self.delete_intervention),
                   ('Undo', 'Ctrl+Z', self.undo),
                   ('Delete entrapment', '', self.delete_entrapment)]
        grid = QGridLayout()
        for k, (text, key, f) in enumerate(actions):
            b = QPushButton(f'{text} ({key})' if key and key != 'Ctrl+Z' else text)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(f)
            grid.addWidget(b, k // 2, k % 2)
            if key and key != 'Ctrl+Z':
                QShortcut(QKeySequence(key), self, f)

        self.types = {}
        for k, t in enumerate(TYPES):
            self.types[t] = QCheckBox(f'{t} ({k + 1})')
            QShortcut(QKeySequence(str(k + 1)), self, lambda t=t: self.toggle(t))
        self.fall = QCheckBox('fall (F)')
        QShortcut(QKeySequence('F'), self, lambda: self.toggle('fall'))
        for t, box in [*self.types.items(), ('fall', self.fall)]:
            box.setFocusPolicy(Qt.NoFocus)
            box.clicked.connect(lambda _, t=t: self.toggle(t))

        self.events = QListWidget()
        self.events.itemClicked.connect(lambda _: self.select_row(self.events.currentRow()))
        self.interventions = QListWidget()
        self.candidates = QListWidget()
        if self.compare:
            a, b = self.compare
            spots = labels.disagreements(self.compare[a], self.compare[b], a, b)
            title = f'Where {a} and {b} disagree (click to go to one)'
        else:
            spots = [(a, f'{a:7.1f} - {b:7.1f} s') for a, b in self.walk.candidates]
            title = 'Candidates: behind command or rolled (click to go to one)'
        self.candidates.addItems([text for _, text in spots])
        self.candidates.itemClicked.connect(lambda _: self.jump(spots[self.candidates.currentRow()][0]))
        self.saved = QLabel()

        side = QVBoxLayout()
        side.addWidget(QLabel('Entrapments (click to go to one)'))
        side.addWidget(self.events, 2)
        side.addLayout(grid)
        side.addWidget(QLabel('Entrapment at the cursor:'))
        for box in [*self.types.values(), self.fall]:
            side.addWidget(box)
        side.addWidget(QLabel('Its interventions'))
        side.addWidget(self.interventions, 1)
        side.addWidget(QLabel(title))
        side.addWidget(self.candidates, 1)
        keys = QLabel('Space play / pause; Left, Right one frame, with Shift 1 s. Click a lane to move the cursor. '
                      'Drag along the entrapments lane to draw one; drag a box or its edge to change it.')
        keys.setWordWrap(True)
        side.addWidget(keys)
        side.addWidget(self.saved)
        panel = QWidget()
        panel.setLayout(side)
        panel.setFixedWidth(380)
        return panel

    def build_lanes(self):
        w = self.walk
        lanes = [('entrapments', []), ('interventions', []), ('candidates', [])] + [(n, []) for n in self.compare]
        lanes += [('speed', [(w.cmd_speed, 'commanded m/s', (120, 120, 120), Qt.DashLine),
                             (w.obs_speed, 'measured m/s', (40, 40, 40), Qt.SolidLine)]),
                  ('behind', [(w.behind, 'behind command over 3 s, m', (200, 60, 60), Qt.SolidLine)]),
                  ('legs', [(w.legs, 'sum of joint speeds, rad/s', (60, 120, 60), Qt.SolidLine)]),
                  ('attitude', [(w.roll, 'roll deg', (60, 90, 200), Qt.SolidLine),
                                (w.pitch, 'pitch deg', (200, 120, 40), Qt.SolidLine)]),
                  ('joystick', [(w.stick[:, 1], 'forward', (40, 40, 40), Qt.SolidLine),
                                (w.stick[:, 0], 'sideways', (60, 90, 200), Qt.SolidLine),
                                (w.stick[:, 2], 'turn', (200, 120, 40), Qt.SolidLine)])]
        self.plots, self.cursors = {}, []
        first = None
        for row, (name, curves) in enumerate(lanes):
            p = self.lanes.addPlot(row=row, col=0, viewBox=DragLane(self.draw_new) if row == 0 else None)
            p.getAxis('left').setWidth(90)
            p.hideButtons()
            p.setMouseEnabled(x=True, y=False)
            p.showAxis('bottom', row == len(lanes) - 1)
            if first is None:
                first = p
            else:
                p.setXLink(first)
            if curves:
                p.setLabel('left', name)
                p.addLegend(offset=(5, 2), labelTextSize='8pt', brush=pg.mkBrush(255, 255, 255, 220), colCount=3)
            else:
                p.setYRange(0, 1)
                p.getAxis('left').setTicks([[(0.5, name)]])
                p.setMaximumHeight(42)
            for y, label, colour, style in curves:
                p.plot(w.t, y, pen=pg.mkPen(colour, width=1.2, style=style), name=label, connect='finite')
            cursor = pg.InfiniteLine(pos=0, movable=True, pen=pg.mkPen((0, 0, 0), width=1.5))
            cursor.sigDragged.connect(lambda c: self.set_time(c.value()))
            p.addItem(cursor)
            self.plots[name] = p
            self.cursors.append(cursor)
        threshold = pg.InfiniteLine(pos=BEHIND, angle=0, pen=pg.mkPen((200, 60, 60), style=Qt.DotLine))
        self.plots['behind'].addItem(threshold)
        for a, b in w.unreliable:
            self.plots['speed'].addItem(pg.LinearRegionItem((a, b), movable=False, brush=brush('unreliable', 70)))
        key = pg.PlotDataItem([np.nan], [np.nan], pen=None, fillLevel=0, fillBrush=brush('unreliable', 140))
        self.plots['speed'].legend.addItem(key, 'lidar off or odometry not confirmed')
        for a, b in w.candidates:
            self.plots['candidates'].addItem(pg.LinearRegionItem((a, b), movable=False, brush=brush('candidate', 90)))
        for name, entrapments in self.compare.items():
            for e in entrapments:
                self.plots[name].addItem(pg.LinearRegionItem((e.start, e.end), movable=False, brush=brush('other', 90)))
                text = ', '.join(e.types) + (', fall' if e.fall else '')
                text += ''.join(f' +{i.kind[0].upper()}' for i in e.interventions)
                label = pg.TextItem(text, color=(0, 0, 0), anchor=(0, 0))
                label.setPos(e.start, 1)
                self.plots[name].addItem(label)
        first.setXRange(0, 60)
        self.lanes.scene().sigMouseClicked.connect(self.click)
        self.boxes = []

    def set_time(self, t):
        self.time = float(np.clip(t, 0, self.duration))
        for c in self.cursors:
            c.setValue(self.time)
        view = self.plots['entrapments']
        lo, hi = view.viewRange()[0]
        if not lo <= self.time <= hi:
            view.setXRange(self.time - 0.2 * (hi - lo), self.time + 0.8 * (hi - lo), padding=0)
        self.sticks.show_at(self.walk.stick, int(round(self.time / DT)))
        text = f'{self.walk.seq}   t = {self.time:.2f} s after the bag start'
        if self.video:
            vt = self.walk.video_time(self.time)
            img = self.video.image(vt)
            text += f'   video {vt:.2f} s'
            if img is not None:
                self.frame.setPixmap(QPixmap.fromImage(img).scaled(self.frame.size(), Qt.KeepAspectRatio))
        if any(a <= self.time <= b for a, b in self.walk.unreliable):
            text += '   (measured speed: lidar off or odometry not confirmed)'
        self.status.setText(text)
        here = self.at_cursor()
        if here is not self.shown:
            self.selected = here or self.selected
            self.refresh()

    def tick(self):
        if self.time >= self.duration:
            self.timer.stop()
        self.set_time(self.time + 0.04)

    def jump(self, t):
        self.timer.stop()
        self.set_time(t)

    def play(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(40)

    def click(self, ev):
        if ev.button() == Qt.LeftButton:
            self.jump(self.plots['entrapments'].vb.mapSceneToView(ev.scenePos()).x())

    def at_cursor(self):
        hits = [e for e in self.entrapments if e.start <= self.time <= e.end]
        return self.selected if self.selected in hits else (hits[0] if hits else None)

    def edit(self, change):
        """Apply a change to the labels, keep it for undo, redraw and save."""
        self.history.append(copy.deepcopy((self.entrapments, self.entrapments.index(self.selected)
                                           if self.selected in self.entrapments else None)))
        change()
        self.refresh()
        self.save()

    def undo(self):
        if self.history:
            self.entrapments, k = self.history.pop()
            self.selected = self.entrapments[k] if k is not None else None
            self.refresh()
            self.save()

    def new_entrapment(self):
        def change():
            self.selected = Entrapment(self.time, min(self.time + 2, self.duration))
            self.entrapments.append(self.selected)
        self.edit(change)

    def draw_new(self, a, b):
        def change():
            self.selected = Entrapment(max(a, 0.0), min(b, self.duration))
            self.entrapments.append(self.selected)
        self.edit(change)
        self.jump(a)

    def from_candidate(self):
        hit = [(a, b) for a, b in self.walk.candidates if a <= self.time <= b]
        if hit:
            def change():
                self.selected = Entrapment(*hit[0])
                self.entrapments.append(self.selected)
            self.edit(change)

    def move_edge(self, side):
        e = self.selected
        if e is None or (side == 0 and self.time >= e.end) or (side == 1 and self.time <= e.start):
            return

        def change():
            if side == 0:
                e.start = self.time
            else:
                e.end = self.time
            self.fit_interventions(e)
        self.edit(change)

    def add_intervention(self, kind):
        """From the cursor to the end of the entrapment; a physical one ends a controller one still running."""
        e = self.at_cursor()
        if e is None or self.time >= e.end:
            return

        def change():
            if kind == 'physical':
                for i in e.interventions:
                    if i.kind == 'controller' and i.start < self.time < i.end:
                        i.end = self.time
            e.interventions.append(Intervention(kind, self.time, e.end))
        self.edit(change)

    def end_controller(self):
        e = self.at_cursor()
        own = [i for i in e.interventions if i.kind == 'controller' and i.start < self.time] if e else []
        if own:
            i = max(own, key=lambda i: i.start)
            self.edit(lambda: setattr(i, 'end', self.time))

    def fit_interventions(self, e):
        """Keep interventions inside their entrapment; a physical one lasts to its end."""
        for i in e.interventions:
            i.start = min(max(i.start, e.start), e.end)
            i.end = e.end if i.kind == 'physical' else min(max(i.end, i.start), e.end)

    def toggle(self, flag):
        e = self.at_cursor()
        if e is None:
            self.refresh()
            return

        def change():
            if flag == 'fall':
                e.fall = not e.fall
            elif flag in e.types:
                e.types.remove(flag)
            else:
                e.types.append(flag)
        self.edit(change)

    def select_row(self, row):
        ordered = sorted(self.entrapments, key=lambda e: e.start)
        if 0 <= row < len(ordered):
            self.selected = ordered[row]
            self.jump(self.selected.start)
            self.refresh()

    def delete_intervention(self):
        row, e = self.interventions.currentRow(), self.at_cursor()
        if e is not None and 0 <= row < len(e.interventions):
            self.edit(lambda: e.interventions.pop(row))

    def delete_entrapment(self):
        e = self.at_cursor()
        if e is not None:
            def change():
                self.entrapments.remove(e)
                self.selected = None
            self.edit(change)

    def box_moved(self, box, item, e):
        a, b = box.getRegion()

        def change():
            item.start, item.end = max(a, 0.0), min(b, self.duration)
            self.fit_interventions(e)
            self.selected = e
        QTimer.singleShot(0, lambda: self.edit(change))

    def refresh(self):
        """Redraw the label boxes and the side panel from the labels."""
        for plot, box in self.boxes:
            plot.removeItem(box)
        self.boxes = []
        ordered = sorted(self.entrapments, key=lambda e: e.start)
        for e in ordered:
            alpha = 150 if e is self.selected else 70
            items = [(self.plots['entrapments'], e, 'entrapment')]
            items += [(self.plots['interventions'], i, i.kind) for i in e.interventions]
            for plot, item, colour in items:
                box = pg.LinearRegionItem((item.start, item.end), brush=brush(colour, alpha))
                for line in box.lines:
                    line.setPen(pg.mkPen(*COLOURS[colour], width=3))
                    line.setHoverPen(pg.mkPen(0, 0, 0, width=5))
                box.sigRegionChangeFinished.connect(lambda b, item=item, e=e: self.box_moved(b, item, e))
                plot.addItem(box)
                self.boxes.append((plot, box))

        self.events.clear()
        for k, e in enumerate(ordered):
            flags = ', '.join(e.types) + (', fall' if e.fall else '') or 'no type yet'
            self.events.addItem(f'E{k + 1:03d}  {e.start:.1f} - {e.end:.1f} s  {flags}')
        if self.selected in ordered:
            self.events.setCurrentRow(ordered.index(self.selected))
        e = self.shown = self.at_cursor()
        for t, box in self.types.items():
            box.setEnabled(e is not None)
            box.setChecked(e is not None and t in e.types)
        self.fall.setEnabled(e is not None)
        self.fall.setChecked(e is not None and e.fall)
        self.interventions.clear()
        if e is not None:
            self.interventions.addItems([f'{i.kind}  {i.start:.1f} - {i.end:.1f} s' for i in e.interventions])

    def save(self):
        labels.save(self.path, self.entrapments, self.annotator, self.walk.seq, self.walk.start)
        self.saved.setText(f'Saved to {self.path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('seq')
    ap.add_argument('--annotator', required=True)
    ap.add_argument('--out')
    ap.add_argument('--compare', nargs=2, default=[], metavar='ANNOTATOR')
    a = ap.parse_args()
    compare = {}
    for name in a.compare:
        path = Path(name) if Path(name).exists() else walk_dir(a.root, a.seq) / 'annotations' / f'{name}.yaml'
        compare[Path(name).stem] = labels.load(path)
    app = QApplication(sys.argv)
    pg.setConfigOptions(background='w', foreground='k', antialias=True)
    print(f'loading {a.seq} ...')
    w = Window(Walk(a.root, a.seq), a.out or f'{a.seq}_{a.annotator}.yaml', a.annotator, compare)
    w.resize(1600, 1000)
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
