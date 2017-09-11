#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
~~Pretty fly for a .py-file~~

Fly, a flow visualizer in Python. By Dion Häfner (mail@dionhaefner.de).

Find me on GitHub: https://github.com/dionhaefner/fly
"""

import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from glumpy import app, gl, gloo
from glumpy.geometry.primitives import sphere
from glumpy.transforms import Rotate, Viewport, Position, Translate, Arcball
from glumpy.graphics.text import FontManager
from glumpy.graphics.collections import GlyphCollection, PathCollection, MarkerCollection
from glumpy.app import movie


class Fly:
    """Main class. Used to set up an interactive flow visualization and run it.

    Settings are attributes of this class. The scalar and/or vector fields
    to be visualized are passed to the __init__ constructor.

    Supports showing:
    - scalar fields (shading),
    - vector fields (flow lines),
    - background images,
    - makers / labels
    on an interactive, zoomable sphere.

    Example:
      >>> fly = Fly(scalar_field=temperature, flow_field=velocity)
      >>> fly.rotate = True
      >>> fly.setup()
      >>> fly.run()

    """
    #
    # Settings
    #

    # number of grid lines per dimension
    num_gridlines = (10, 10)

    # matplotlib colormap to use for shading
    colormap = "viridis"

    # background image to project onto the globe; must be on regular lon-lat-grid
    background_image = None

    # window background color
    bgcolor = (0, 0, 0, 1)

    # screen resolution
    resolution = (1920, 1080)

    # number of segments per flow line
    num_segments = 40

    # length of each flow line segment
    segment_length = 1e-2

    # total number of flow lines per dimension
    flow_resolution = (200, 200)

    # apparent speed of the flow lines
    flow_speed = 1e-2

    # mark locations on the map; tuple containing (labels, longitudes, latitudes)
    markers = None

    # make the global auto-rotate
    rotate = False

    # record a video of the animation
    record = False

    #
    # Constants
    #

    _radius = 1.5
    _framerate = 60


    def __init__(self, shading_field=None, flow_field=None):
        self._setup_done = False

        if flow_field is not None:
            flow_field = np.asarray(flow_field)
            if flow_field.ndim != 3 or flow_field.shape[0] != 2:
                raise ValueError("flow_field must be of shape (2, nx, ny)")
        self.flow_field = flow_field / np.nanmax(flow_field)

        if shading_field is not None:
            shading_field = np.asarray(shading_field)
            if shading_field.ndim != 2:
                raise ValueError("shading_field must be two-dimensional")
        shading_field -= np.nanmin(shading_field)
        shading_field /= np.nanmax(shading_field)
        self.shading_field = shading_field


    @staticmethod
    def spheric_to_cartesian(phi, theta, rho):
        """Convert spherical to Cartesian coordinates"""
        x = np.sin(theta) * np.sin(phi) * rho
        y = np.sin(theta) * np.cos(phi) * rho
        z = np.cos(theta) * rho * np.ones_like(x)
        return np.array([x, y, z]).T


    @staticmethod
    def distribute_points_on_sphere(nx, ny):
        """Evenly distribute a number of points on the surface of a sphere"""
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        sx = (x + .5) / nx
        sy = 1 / np.pi * np.arcsin(2 * ((y + 0.5) / ny - 0.5)) + .5
        return sx, sy


    def setup(self):
        self._transform = Arcball(Position(), znear=1, zfar=10, aspect=1)
        self._viewport = Viewport()

        if self.background_image is not None:
            image_data = np.array(Image.open(self.background_image))
            earth_vertices, self._earth_indices = sphere(self._radius, 32, 32)
            self._earth = gloo.Program(ImageShader.vertex, ImageShader.fragment)
            self._earth.bind(earth_vertices)

            self._earth["texture"] = image_data
            self._earth["texture"].interpolation = gl.GL_NEAREST
            self._earth["transform"] = self._transform
        else:
            self._earth = Dummy()
            self._earth_indices = None

        if self.shading_field is not None:
            flipped_field = np.flipud(self.shading_field)
            overlay_vertices, self._overlay_indices = sphere(self._radius * 1.01, 52, 52)
            self._overlay = gloo.Program(ImageShader.vertex, ImageShader.fragment)
            self._overlay.bind(overlay_vertices)
            colors = plt.get_cmap(self.colormap)(flipped_field)
            colors[..., 3] = 0.7 + 0.3 * flipped_field
            self._overlay["texture"] = colors
            self._overlay["texture"].interpolation = gl.GL_LINEAR
            self._overlay["transform"] = self._transform
        else:
            self._overlay = Dummy()
            self._overlay_indices = None

        if any(self.num_gridlines):
            self._paths = PathCollection(mode="agg+", color="global", linewidth="global",
                                   viewport=self._viewport, transform=self._transform)
            self._paths["color"] = 0,0,0,0.5
            self._paths["linewidth"] = 1.0

            theta = np.linspace(0, 2*np.pi, 64, endpoint=True)
            for phi in np.linspace(0, np.pi, self.num_gridlines[0], endpoint=True):
                self._paths.append(self.spheric_to_cartesian(phi, theta, self._radius * 1.011), closed=True)

            phi = np.linspace(0, 2*np.pi, 64, endpoint=True)
            for theta in np.linspace(0, np.pi, self.num_gridlines[1], endpoint=True)[1:-1]:
                self._paths.append(self.spheric_to_cartesian(phi, theta, self._radius * 1.011), closed=True)
        else:
            self._paths = Dummy()

        if self.markers is not None:
            self._markers = MarkerCollection(marker="disc",
                                             vertex=MarkerShader.vertex,
                                             viewport=self._viewport,
                                             transform=self._transform)
            labels, latitudes, longitudes = self.markers
            points = self.spheric_to_cartesian(np.pi + np.pi / 180. * np.asarray(longitudes),
                                               np.pi / 2 + np.pi / 180. * np.asarray(latitudes),
                                               self._radius * 1.01)
            self._markers.append(points, bg_color=(1,1,1,1), fg_color=(.25,.25,.25,1), size=10)

            self._labels = GlyphCollection("agg", vertex=GlyphShader.vertex,
                                           transform=self._transform,
                                           viewport=self._viewport)
            font = FontManager.get("OpenSans-Regular.ttf", size=16, mode="agg")
            for label, p in zip(labels, points):
                self._labels.append(label, font, origin=p, color=(1,1,1,1))
            self._labels["position"][:,1] -= 20
        else:
            self._markers = Dummy()
            self._labels = Dummy()

        if self.flow_field is not None:
            field = np.transpose(self.flow_field, (1, 2, 0)).astype(np.float32).copy()

            rows, cols = self.flow_resolution
            p_x, p_y = self.distribute_points_on_sphere(rows, cols)
            self._flow_indices = (rows * p_x + rows * cols * p_y).flatten()

            self._flow = gloo.Program(FlowLineShader.vertex, FlowLineShader.fragment)
            index = np.empty((rows * cols, self.num_segments * 2, 2), dtype=np.float32)
            index[:, :, 0] = self._flow_indices[:, np.newaxis]
            index[:, ::2, 1] = np.arange(self.num_segments)[np.newaxis, :]
            index[:, 1::2, 1] = np.arange(self.num_segments)[np.newaxis, :] + 1
            self._flow["index"] = index
            self._flow["field"] = field
            self._flow["field"].interpolation = gl.GL_LINEAR
            self._flow["field"].gpu_format = gl.GL_RG32F
            self._flow["field"].wrapping = gl.GL_MIRRORED_REPEAT
            self._flow["field_shape"] = (rows, cols)
            self._flow["offset"] = np.random.uniform(0., 1E-2, size=(rows, cols, 3)).astype(np.float32)
            self._flow["speed"] = self.flow_speed
            self._flow["color"] = np.reshape([1,1,1,.5],(1,1,4)).astype(np.float32)
            self._flow["seg_len"] = self.segment_length
            self._flow["nseg"] = self.num_segments
            self._flow["time"] = 0
            self._flow["radius"] = self._radius * 1.02
            self._flow["transform"] = self._transform
        else:
            self._flow = Dummy()
            self._flow_indices = None

        self._setup_done = True


    def run(self):
        if not self._setup_done:
            raise RuntimeError("must call setup() method before running")

        config = app.configuration.Configuration()
        config.samples = 2
        window = app.Window(width=self.resolution[0] * 2, height=self.resolution[1] * 2,
                            color=self.bgcolor, fullscreen=True,
                            decoration=False, config=config)
        window.attach(self._transform)
        window.attach(self._viewport)

        @window.event
        def on_draw(dt):
            window.clear()
            gl.glEnable(gl.GL_DEPTH_TEST)

            if self.rotate:
                self._transform.theta += .1
            self._earth.draw(gl.GL_TRIANGLES, self._earth_indices)
            self._paths.draw()
            self._overlay.draw(gl.GL_TRIANGLES, self._overlay_indices)
            self._flow.draw(gl.GL_LINES, self._flow_indices)
            self._flow["time"] += 1
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_BLEND)

            self._markers.draw()
            self._labels.draw()

        @window.event
        def on_init():
            gl.glEnable(gl.GL_DEPTH_TEST)
            self._transform.phi = 125
            self._transform.theta = -150
            self._transform.zoom = 30
            self._transform.distance = 8

        if self.record:
            del sys.argv[2:]
            filename = "%s.mp4" % time.strftime("%Y-%m-%d_%H-%M-%S")
            with movie.record(window, filename, fps=self._framerate):
                app.run(framerate=self._framerate)
            print("movie saved as %s" % filename)
        else:
            app.run(framerate=self._framerate)


class Dummy:
    def draw(*args, **kwargs):
        pass


"""
Shaders
"""

class ImageShader:
    vertex = """
    uniform mat4 model, view, projection;
    attribute vec3 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        v_texcoord  = texcoord;
        gl_Position = <transform(position)>;
    }
    """

    fragment = """
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        vec4 v = texture2D(texture, v_texcoord);
        gl_FragColor = v;
    }
    """


class MarkerShader:
    vertex = """
    #include "math/constants.glsl"

    varying float v_size;
    varying vec4  v_fg_color;
    varying vec4  v_bg_color;
    varying vec2  v_orientation;
    varying float v_antialias;
    varying float v_linewidth;
    void main (void)
    {
        fetch_uniforms();
        v_linewidth   = linewidth;
        v_antialias   = antialias;
        v_fg_color    = fg_color;
        v_bg_color    = bg_color;
        v_orientation = vec2(cos(orientation), sin(orientation));

        gl_Position = <transform(position)>;
        float scale = (3.5 - length(gl_Position.xyz)/length(vec3(1.5)));
        v_fg_color.a = scale;
        v_bg_color.a = scale;
        v_size       = size * scale;
        gl_PointSize = M_SQRT2 * size * scale + 2.0 * (linewidth + 1.5*antialias);
        <viewport.transform>;
    }
    """


class GlyphShader:
    vertex = """
    varying vec4  v_color;
    varying float v_offset;
    varying vec2  v_texcoord;

    void main()
    {
        fetch_uniforms();

        gl_Position = <transform(origin)>;
        v_color = color;
        v_texcoord = texcoord;
        <viewport.transform>;

        float scale = (3.5 - length(gl_Position.xyz)/length(vec3(1.5)));
        v_color.a = scale;

        // We set actual position after transform
        v_offset = 3.0*(offset + origin.x - int(origin.x));
        gl_Position /= gl_Position.w;
        gl_Position = gl_Position + vec4(2.0*position/<viewport.viewport_global>.zw,0,0);
    }
    """


class FlowLineShader:
    vertex = """
    attribute vec2 index;
    uniform sampler2D field;
    uniform sampler2D color;
    uniform sampler2D offset;
    uniform vec2 field_shape;
    uniform float radius;
    uniform float seg_len;
    varying vec4 base_color;
    varying vec2 texcoord;
    varying float dist;

    vec4 to_cartesian(vec2 unit_pos) {
        float pi = 3.141592;
        vec2 latlon;
        // lon from 0 to 2pi, lat from 0 to pi
        latlon.x = 2*pi*unit_pos.x;
        latlon.y = pi*unit_pos.y;
        float x = sin(latlon.x) * sin(latlon.y) * radius;
        float y = cos(latlon.x) * sin(latlon.y) * radius;
        float z = cos(latlon.y) * radius;
        return vec4(x,y,z,1.0);
    }

    vec2 rk4(vec2 tpos, float dx) {
        vec2 uv1, uv2, uv3, uv4;
        vec2 dir1, dir2, dir3, dir4;
        uv1 = tpos;
        dir1 = texture2D(field, uv1).xy;
        uv2 = tpos + dx/2 * dir1;
        dir2 = texture2D(field, uv2).xy;
        uv3 = tpos + dx/2 * dir2;
        dir3 = texture2D(field, uv3).xy;
        uv4 = tpos + dx * dir3;
        dir4 = texture2D(field, uv4).xy;
        return tpos + dx/6 * (dir1 + 2*dir2 + 2*dir3 + dir4);
    }

    void main() {
        dist = index.y * seg_len;
        texcoord = vec2(mod(index.x, field_shape.x), floor(index.x / field_shape.x)) / field_shape;
        vec2 off = texture2D(offset, texcoord).xy;
        vec2 local = mod(texcoord + off, 1);
        for(int i=0; i < index.y; i+=1) {
            local = rk4(local, seg_len);
        }
        base_color = texture2D(color, local);

        gl_Position = <transform(to_cartesian(local))>;
    }
    """

    fragment = """
    uniform sampler2D offset;
    uniform float time;
    uniform float speed;
    uniform float nseg;
    uniform float seg_len;
    varying vec4 base_color;
    varying vec2 texcoord;
    varying float dist;

    void main() {
        float totlen = nseg * seg_len;
        float phase = texture2D(offset, texcoord).b * 100;
        float alpha;

        // vary alpha along the length of the line to give the appearance of motion
        alpha = mod((dist / totlen) + phase - time * speed, 1);

        // add a cosine envelope to fade in and out smoothly at the ends
        alpha *= (1 - cos(2 * 3.141592 * dist / totlen)) * 0.5;

        gl_FragColor = vec4(base_color.rgb, base_color.a * alpha);
    }
    """
