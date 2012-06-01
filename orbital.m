% Orbital.m

G = 6.6738480e-11;
M = 5.9742e24;
mu = G * M;

v = [100.0 -10.0];
r = [100000.0 60000.0];
r_mag = norm(r);
r_dir = r./r_mag;
vr = r_dir .* dot(r_dir, v); % radial velocity
vr_mag = norm(vr);
vt = v - vr; % tangent velocity
vt_mag = norm(vt);

t_dir = vt./vt_mag;

% Compute Kepler orbit

p = ((r_mag .* vt_mag) ^ 2) ./ mu;
v0 = sqrt(mu/p);

ex = ((vt_mag - v0) * r_dir - vr_mag * t_dir)/v0;
e = norm(ex);

ec = (vt_mag / v0) - 1;
es = (vr_mag / v0);
theta = atan2(es, ec);

x_dir = cos(theta) * r_dir - sin(theta) * t_dir;
y_dir = sin(theta) * r_dir + cos(theta) * t_dir;

xs = [];
ys = [];
steps = 10000;
% e = 2.0; % TODO 1.0 sometimes works, > 1 doesn't - do we need to just
% restrict the range of theta?
delta = 0.0001;
HAX_RANGE = 0.9; % limit range to stay out of very large values
% TODO want to instead limit the range based on... some viewing area?
% might be two visible segments, one from +ve and one from -ve theta, with
% different visible ranges. Could determine 
% TODO and want to take steps of fixed length/distance
if (e < 1 - delta) % ellipse
    range = pi;
elseif (e < 1 + delta) % parabola
    range = pi * HAX_RANGE;
else % hyperbola
    range = acos(-1/e) * HAX_RANGE;
end
mint = -range;
maxt = range;
for i = 0:steps
    ct = ((i / steps) * (maxt - mint)) + mint;
    cr = p / (1 + e * cos(ct));
    
    % TODO this part seems broken?
    x_len = cr .* cos(ct);
    y_len = cr .* sin(ct);
    pos = (x_dir .* x_len) + (y_dir .* y_len);
    xs = [xs pos(1)];
    ys = [ys pos(2)];
    % xs = [xs x_len];
    % ys = [ys y_len];
end

plot(xs, ys, '+');
hold on;
plot(r(1), r(2), 'r+');