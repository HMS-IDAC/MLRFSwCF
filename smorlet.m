% Generates 2D real and imaginary Morlet wavelet kernels
%
% MORLET WAVELET (according to Wikipedia, as of Aug 16 2012)
% The Morlet wavelet (or Gabor wavelet) is a wavelet
% composed of a complex exponential (carrier) multiplied by
% a Gaussian window (envelope).
% This wavelet is closely related to human perception,
% both hearing and vision.
%
% USAGE
% [mr,mi] = smorlet(stretch,scale,orientation,npeaks);
%
% RETURNS
% mr: real part of kernel (in the range [-1,1])
% mi: imaginary part of kernel (in the range [-1,1])
%
% PARAMETERS
% stretch: controls how stretched is the wavelet
%        typical values: 1,2,...,10
% scale: controls the size of the kernel
%        typical values: 1,2,...,20
% orientation: angle of rotation, in degrees
%        typical values: anything in the range [0,360)
% npeaks: rough number of significant peaks appearing in the kernel
%        typical values: 1,2,...,10
%
% EXAMPLE
% scale = 20;
% orientation = 45;
% npeaks = 3;
% stretch = 3;
% 
% [mr,mi] = smorlet(stretch,scale,orientation,npeaks);
% 
% mr = mr-min(min(mr));
% mr = mr/max(max(mr));
% imshow(mr)
% 
% mi = mi-min(min(mi));
% mi = mi/max(max(mi));
% figure
% imshow(mi)
%
% VERSION
% 1.0, Mar 26 2013
%
% AUTHOR
% Marcelo Cicconet, New York University
% marceloc.net

function [mr,mi] = smorlet(stretch,scale,orientation,npeaks)

% controls width of gaussian window (default: scale)
sigma = scale;

% orientation (in radians)
theta = -(orientation-90)/360*2*pi;

% controls elongation in direction perpendicular to wave
gamma = 1/(1+stretch);

% width and height of kernel
support = ceil(2.5*sigma/gamma);

% wavelength (default: 4*sigma)
lambda = 1/npeaks*4*sigma;

% phase offset (in radians)
psi = 0;


xmin = -support;
xmax = -xmin;
ymin = xmin;
ymax = xmax;

xdomain = xmin:xmax;
ydomain = ymin:ymax;

[x,y] = meshgrid(xdomain,ydomain);

xprime = cos(theta)*x+sin(theta)*y;
yprime = -sin(theta)*x+cos(theta)*y;

expf = exp(-0.5/sigma^2*(xprime.^2+gamma^2*yprime.^2));

mr = expf.*cos(2*pi/lambda*xprime+psi);
mi = expf.*sin(2*pi/lambda*xprime+psi);

% mean = 0
mr = mr-sum(sum(mr))/numel(mr);
mi = mi-sum(sum(mi))/numel(mi);

% norm = 1
mr = mr./sqrt(sum(sum(mr.*mr)));
mi = mi./sqrt(sum(sum(mi.*mi)));

end