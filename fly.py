"""
Pretty fly for a Py-file!
"""

import csv
import numpy as np
import pygrib
from PIL import Image
from glumpy import app, gl, glm, gloo, data
app.use("freeglut")
from glumpy.geometry.primitives import sphere
from glumpy.transforms import Rotate, Viewport, Position, Translate, Arcball
from glumpy.graphics.text import FontManager
from glumpy.graphics.collections import GlyphCollection, PathCollection, MarkerCollection


def shiftdata(lon,arr):
    ix = np.argmin(np.abs(lon-180), axis=1)
    assert (len(set(ix)) == 1)
    ix = ix[0]
    slon = np.hstack((lon[:,ix:] - 360., lon[:,:ix]))
    sarr = np.hstack((arr[:,ix:], arr[:,:ix]))
    assert (slon.shape == lon.shape)
    assert (sarr.shape == arr.shape)
    return slon, sarr

def read_wind_data(path):
    grbs = pygrib.open(path)
    grbs.seek(0)
    grb = grbs.readline()
    lat0, lon0 = grb.latlons()
    u0 = np.flipud(grb.values)
    grb = grbs.readline()
    v0 = np.flipud(grb.values)
    grbs.close()
    lat0 = np.flipud(lat0)
    lon, lat = shiftdata(lon0,lat0)
    u = shiftdata(lon0,u0)[1]
    v = shiftdata(lon0,v0)[1]
    return u, v

def spheric_to_cartesian(phi, theta, rho):
    """ Spheric to cartesian coordinates """
    if   hasattr(phi, '__iter__'):   n = len(phi)
    elif hasattr(theta, '__iter__'): n = len(theta)
    elif hasattr(rho, '__iter__'):   n = len(rho)
    P = np.empty((n,3), dtype=np.float32)
    sin_theta = np.sin(theta)
    P[:,0] = sin_theta * np.sin(phi) * rho
    P[:,1] = sin_theta * np.cos(phi) * rho
    P[:,2] = np.cos(theta) * rho
    return P


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

transform = Arcball(Position(),znear=1,zfar=10,aspect=1)
viewport  = Viewport()

radius = 1.5
vertices, indices = sphere(radius, 32, 32)
earth = gloo.Program(vertex, fragment)
earth.bind(vertices)
earth['texture'] = np.array(Image.open("bluemarble.jpg"))
earth['texture'].interpolation = gl.GL_LINEAR
earth['transform'] = transform

paths = PathCollection(mode="agg+", color="global", linewidth="global",
                       viewport=viewport, transform=transform)
paths["color"] = 0,0,0,0.5
paths["linewidth"] = 1.0

theta = np.linspace(0, 2*np.pi, 64, endpoint=True)
for phi in np.linspace(0, np.pi, 12, endpoint=True):
    paths.append(spheric_to_cartesian(phi, theta, radius*1.01), closed=True)

phi = np.linspace(0, 2*np.pi, 64, endpoint=True)
for theta in np.linspace(0, np.pi, 19, endpoint=True)[1:-1]:
    paths.append(spheric_to_cartesian(phi, theta, radius*1.01), closed=True)

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

markers = MarkerCollection(marker="disc", vertex=vertex,
                           viewport = viewport, transform=transform)
C, La, Lo = [], [], []
with open(data.get("capitals.csv"), 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader, None) # skip the header
    for row in reader:
        capital = row[1]
        latitude = np.pi/2 + float(row[2])*np.pi/180
        longitude = np.pi  + float(row[3])*np.pi/180
        C.append(capital)
        La.append(latitude)
        Lo.append(longitude)
P = spheric_to_cartesian(Lo, La, radius*1.01)
markers.append(P, bg_color = (1,1,1,1), fg_color=(.25,.25,.25,1), size = 10)


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

labels = GlyphCollection('agg', vertex=vertex,
                         transform=transform, viewport=viewport)
font = FontManager.get("OpenSans-Regular.ttf", size=16, mode='agg')
for i in range(len(P)):
    labels.append(C[i], font, origin = P[i], color=(1,1,1,1))
labels["position"][:,1] -= 20


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

u, v = read_wind_data("wind_data.grib")
field = np.dstack((u,v)).astype(np.float32)

def distribute_points(nx,ny):
    x, y = np.meshgrid(np.arange(nx),np.arange(ny))
    sx = (x + .5) / nx
    sy = 1/np.pi * np.arcsin(2*((y+0.5)/ny-0.5)) + .5
    return sx, sy

rows, cols = (100, 100)
p_x, p_y = distribute_points(rows, cols)
vertex_indices = (rows * p_x + rows * cols * p_y).flatten()

segments = 20
flow = gloo.Program(vertex, fragment)
index = np.empty((rows * cols, segments * 2, 2), dtype=np.float32)
index[:, :, 0] = vertex_indices[:, np.newaxis]
index[:, ::2, 1] = np.arange(segments)[np.newaxis, :]
index[:, 1::2, 1] = np.arange(segments)[np.newaxis, :] + 1
flow['index'] = index
flow['field'] = field
flow['field'].interpolation = gl.GL_LINEAR
flow['field'].gpu_format = gl.GL_RG32F
flow['field'].wrapping = gl.GL_MIRRORED_REPEAT
flow['field_shape'] = (rows, cols)
flow['offset'] = np.random.uniform(0., 1E-2, size=(rows, cols, 3)).astype(np.float32)
flow['speed'] = 0.01
flow['color'] = np.reshape([1,1,1,.5],(1,1,4)).astype(np.float32)
flow['seg_len'] = 2E-4
flow['nseg'] = segments
flow['time'] = 0
flow['radius'] = radius * 1.02
flow['transform'] = transform

config = app.configuration.Configuration()
config.samples = 2
window = app.Window(width=1920*2, height=1080*2, color=(0,0,0,0.8), fullscreen=True,
                    decoration=False, config=config)
window.attach(transform)
window.attach(viewport)

@window.event
def on_draw(dt):
    window.clear()
    gl.glEnable(gl.GL_DEPTH_TEST)
    earth.draw(gl.GL_TRIANGLES, indices)
    transform.theta += .1
    paths.draw()
    flow.draw(gl.GL_LINES, range(rows*cols))
    flow["time"] += 1
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    markers.draw()
    labels.draw()

@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    transform.phi = 125
    transform.theta = -150
    transform.zoom = 30
    transform.distance = 8

app.run()
