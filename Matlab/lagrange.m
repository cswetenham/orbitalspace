G = 6.6738480e-11; % m^3 kg^-1 s^-2
M1 = 5.9742e24; % kg
M2 = 7.3477e22; % kg
R = 384400000; % m

Mu = G*(M1 + M2); % Standard gravitational parameter
AngVelSq = Mu / (R^3); % Square of angular velocity of orbit

% Mass 1 at 0,0
% Mass 2 at R,0

% Center of mass of 2-body system
CMX = (0 * M1 + R * M2) / (M1 + M2);
CMY = (0 * M1 + 0 * M2) / (M1 + M2);

figure
% all x,y pairs in the grid
[Xs,Ys] = meshgrid(-1.5*R:10000000:1.5*R);
% vector to mass 1
DX1s = (Xs - 0);
DY1s = (Ys - 0);
% squared distances to mass 1
D1s = DX1s.^2 + DY1s.^2;
% unit directions to mass 1
NX1s = -DX1s ./ sqrt(D1s);
NY1s = -DY1s ./ sqrt(D1s);
% acceleration magnitude towards mass 1
A1s = G * M1 ./ D1s;
% acceleration vector towards mass 1
AX1s = NX1s .* A1s;
AY1s = NY1s .* A1s;

% vector to mass 2
DX2s = (Xs - R);
DY2s = (Ys - 0);
% squared distances to mass 2
D2s = DX2s.^2 + DY2s.^2;
% unit directions to mass 2
NX2s = -DX2s ./ sqrt(D2s);
NY2s = -DY2s ./ sqrt(D2s);
% acceleration magnitude towards mass 2
A2s = G * M2 ./ D2s;
% acceleration vector towards mass 2
AX2s = NX2s .* A2s;
AY2s = NY2s .* A2s;

% Centripetal accelaration in the rotating reference frame
% (at every point, acceleration required to match the period
% of the two bodies' orbit)
DCMXs = (Xs - CMX);
DCMYs = (Ys - CMY);

ACXs = DCMXs .* AngVelSq;
ACYs = DCMYs .* AngVelSq;

AXs = AX1s + AX2s + ACXs;
AYs = AY1s + AY2s + ACYs;

%AXs = ACXs;
%AYs = ACYs;

% Hack - cap magnitudes
MaxMag = 0.001;
AMs = sqrt(AXs.^2 + AYs.^2);
Over = (AMs > MaxMag);
Scales = (Over .* (MaxMag ./ AMs) + (1 - Over));
AXs = Scales .* AXs;
AYs = Scales .* AYs;

% Hack - log-scale vector lengths
%Ls = sqrt(AXs.^2 + AYs.^2);
%LLs = log(Ls + 1);
%Ss = LLs ./ Ls;
%AXs = AXs .* Ss;
%AYs = AYs .* Ss;

%contour(Xs,Ys,Mags)
%hold on
quiver(Xs,Ys,AXs,AYs, 1)
colormap hsv
hold off