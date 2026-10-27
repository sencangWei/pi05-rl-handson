"""Viewer-only end-effector rotation guide for the manual recovery collector."""
import numpy as np

COLORS = ((1., .12, .12, 1.), (.12, 1., .12, 1.), (.15, .4, 1., 1.))
AXES = "XYZ"
ROTATION_PRESETS = {"普通旋转": .3, "精细旋转": .1}


class RotationSpeedMixin:
    def _rotation_preset(self, mode):
        if mode not in ROTATION_PRESETS:
            raise ValueError("请选择普通旋转或精细旋转")
        self.translation_joystick.stop()
        self.rotation_mode_var.set(mode)

    def _manual_jog(self, axis, sign):
        if axis >= 3:
            self.lab.manual_jog(axis, sign, scale=ROTATION_PRESETS[self.rotation_mode_var.get()],
                                repeats=int(self.jog_repeats_var.get()))
        else:
            super()._manual_jog(axis, sign)


def rotation_arc(origin, axis, sign, radius=.095):
    """Points following the right-hand rule about a WORLD axis through the EEF."""
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all() or axis not in range(3) or sign not in (-1, 1):
        raise ValueError("Invalid rotation guide")
    u, v = np.eye(3)[(axis + 1) % 3], np.eye(3)[(axis + 2) % 3]
    angles = sign * np.linspace(0., 1.5 * np.pi, 25)
    return origin + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)


def paint_rotation(scene, origin, axis, sign):
    """Append annotations to user_scn; never change simulation/model/camera images."""
    import mujoco
    points = rotation_arc(origin, axis, sign)
    start = scene.ngeom
    if start + len(points) + 2 > scene.maxgeom:
        raise ValueError("Viewer scene has no space for rotation annotations")
    color = np.asarray(COLORS[axis], dtype=np.float32)
    for i in range(len(points) - 1):
        geom = scene.geoms[start + i]
        kind = mujoco.mjtGeom.mjGEOM_ARROW if i == len(points) - 2 else mujoco.mjtGeom.mjGEOM_LINE
        mujoco.mjv_initGeom(geom, kind, np.zeros(3), points[i], np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, kind, .008 if kind == mujoco.mjtGeom.mjGEOM_ARROW else 3.,
                            points[i], points[i + 1])
        geom.label = ""
    index = start + len(points) - 1
    # A straight axis through the real control point identifies the rotation pivot.
    geom = scene.geoms[index]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), origin,
                       np.eye(3).ravel(), color)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .004,
                        np.asarray(origin) - np.eye(3)[axis] * .13,
                        np.asarray(origin) + np.eye(3)[axis] * .16)
    geom.label = ""
    label = scene.geoms[index + 1]
    mujoco.mjv_initGeom(label, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                       np.asarray(origin) + np.eye(3)[axis] * .19, np.eye(3).ravel(), color)
    label.label = f"R{AXES[axis].lower()}{'+' if sign > 0 else '-'} (world {AXES[axis]})"
    scene.ngeom = index + 2


def guided_classes():
    """Import the pinned UI only after the launcher verifies its hashes."""
    import tkinter as tk
    from tkinter import ttk
    from lab_panel import LabPanel
    from live_preview import LivePreview
    from viewer_controls import paint_world_axes

    class GuidedPreview(LivePreview):
        rotation_axis, rotation_sign = 2, 1

        def sync(self):
            with self.handle.lock():
                origin = self.env.sim.data.site_xpos[self._eef_site_id].copy()
                paint_world_axes(self.handle.user_scn, origin)
                paint_rotation(self.handle.user_scn, origin, self.rotation_axis, self.rotation_sign)
            self.handle.sync()

        def show_rotation(self, axis, sign):
            self.rotation_axis, self.rotation_sign = axis, sign
            self.sync()

    class GuidedPanel(RotationSpeedMixin, LabPanel):
        def __init__(self, root, lab):
            super().__init__(root, lab)
            options = self.normal_motion_button.master
            for widget in options.grid_slaves(row=2):
                widget.grid_configure(row=3)
            self.rotation_mode_var = tk.StringVar(value="精细旋转")
            self.rotation_buttons = {}
            for column, (mode, scale) in enumerate(ROTATION_PRESETS.items()):
                button = ttk.Radiobutton(options, text=f"{mode}（{scale}）", value=mode,
                    variable=self.rotation_mode_var, command=lambda m=mode: self._rotation_preset(m))
                button.grid(row=2, column=column * 2, columnspan=2, sticky="w")
                self.rotation_buttons[mode] = button
            self.motion_hint_var.set("平移：普通 0.6／精细 0.1；旋转单独选档，默认精细。")
            self.rotation_hint = tk.StringVar(value="鼠标指向旋转键：三维窗口显示对应旋转箭头；点击才会运动。")
            controls = self.jog_buttons[(3, 1)].master.master
            text = ("旋转控制夹爪整体姿态，多个关节协同，不是某一节臂的开关。\n"
                    "Rx 绕红 X 轴倾斜；Ry 绕绿 Y 轴倾斜；Rz 绕蓝 Z 轴水平转向。\n"
                    "＋沿弧形箭头转；−反向转。箭头过夹爪控制点；世界轴不随视角转动。")
            ttk.Label(controls, text=text, wraplength=720, justify="left").grid(
                row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
            ttk.Label(controls, textvariable=self.rotation_hint, wraplength=720,
                      justify="left").grid(row=5, column=0, columnspan=2, sticky="w")
            style = ttk.Style(root)
            for axis, color in enumerate(("#c82020", "#187d25", "#235ac4")):
                name = f"Rotation{axis}.TButton"
                style.configure(name, foreground=color)
                for sign in (-1, 1):
                    button = self.jog_buttons[(axis + 3, sign)]
                    button.configure(text=f"R{AXES[axis].lower()} {'+' if sign > 0 else '−'}", style=name)
                    button.bind("<Enter>", lambda event, a=axis, s=sign: self._show_rotation(a, s))
                    button.bind("<FocusIn>", lambda event, a=axis, s=sign: self._show_rotation(a, s))

        def _show_rotation(self, axis, sign):
            if self.lab.view is not None:
                self.lab.view.show_rotation(axis, sign)
            name = f"R{AXES[axis].lower()}{'+' if sign > 0 else '−'}"
            self.rotation_hint.set(f"预览 {name}：夹爪绕世界 {AXES[axis]} 轴按图中箭头转；预览不录动作。")

        def _manual_jog(self, axis, sign):
            if axis >= 3:
                self._show_rotation(axis - 3, sign)
            super()._manual_jog(axis, sign)

    return GuidedPreview, GuidedPanel
