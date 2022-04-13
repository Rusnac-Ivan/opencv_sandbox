

float flare( vec2 U )                            // rotating hexagon 
{	vec2 A = sin(vec2(0, 1.57) + 0.0);
    U = abs( U * mat2(A, -A.y, A.x) ) * mat2(2,0,1,1.7); 
    return .2/max(U.x,U.y);                      // glowing-spiky approx of step(max,.2)
  //return .2*pow(max(U.x,U.y), -2.);
 
}

//#define r(x)     fract(1e4*sin((x)*541.17))      // rand, signed rand   in 1, 2, 3D.
//#define sr2(x)   ( r(vec2(x,x+.1)) *2.-1. )
//#define sr3(x)   ( r(vec4(x,x+.1,x+.2,0)) *2.-1. )

vec2 r(vec2 x)
{
    return fract(1e4 * sin((x) * 541.17));
}

vec4 r1(vec4 x)
{
    return fract(1e4 * sin((x) * 541.17));
}

vec2 sr2(float x)
{
    return r(vec2(x, x + .1)) * 2. - 1.;
}

vec4 sr3(float x)
{
    return r1(vec4(x, x + .1, x + .2, 0)) * 2. - 1.;
}

vec4 stars( vec4 O, vec2 U )
{
    vec2 R = iResolution.xy;
    U =  (U+U - R) / R.y;
	O -= O+.3;
    for (float i=0.; i<99.; i++)
        O += flare (U )//- sr2(i)*R/R.y )           // rotating flare at random location
              * r1(vec4(i+.2))                          // random scale
              * (1.+sin(iDate.w+r1(vec4(i+.3))*6.))*.1  // time pulse
            * (1.+.1*sr3(i+.4))               // random color - uncorrelated
              * (1.+.1*sr3(i));                  // random color - correlated
    return O;
}


#define NB_STARS 200
#define PERS 1          // perspective

#define SCALE 40.
const float star_luminosity = 1e3;
vec3 star_color = vec3(1.,.3,.1)*star_luminosity;
#define PI 3.1415927
vec2 FragCoord, R;

//--- filter integration (l0..l1) on black body spectrum(T) ---------
float F(float x) 
{ return (6.+x*(6.+x*(3.+x)))*exp(-x); }
float IntPlanck(float T,float lambda1,float lambda0) 
{
	const float A=1.1, B=1./1.05;
	float C0 = 0.014387770, C=C0/(B*T);
	T = 1.; // normalised spectrum better for display :-)
	return 100.*A/B*pow(100.*T/C0,4.)*( F(C/lambda1) - F(C/lambda0) );
}

// --- Planck black body color I.spectrum(Temp) -----------------------
vec3 Planck(float T) {
	return vec3(
		IntPlanck(T,.7e-6,.55e-6),   // red filter
        IntPlanck(T,.55e-6,.49e-6),  // green filter
        IntPlanck(T,.49e-6,.4e-6)    // blue filter
		)*1e-14;
}


//--- draw one star:  (I.filter(color)).dirac * PSF ------------------ 
vec3 draw_star(vec2 pos, float I) {
	// star out of screen
    const float margin = .2;
	if (pos!=clamp(pos,vec2(-margin),R/R.y+margin)) return vec3(0.);
	
	pos -= FragCoord.xy/iResolution.y; 
	
// Airy spot = (2BesselJ(1,x)/x)^2 ~ cos^2(x-2Pi/4)/x^3 for x>>1
// pixels >> fringes -> smoothed Airy ~ 1/x^3
	float d = length(pos)*SCALE;
	
	vec3 col, spectrum = I*star_color;
#if 1
	col = spectrum/(d*d*d);
#else
	col = spectrum*(1.+.323*cos(d/4.+PI/2.))/(d*d*d);
#endif
	
// 2ndary mirror handles signature (assuming handles are long ellipses)
	d = length(pos*vec2(50.,.5))*SCALE;
	col += spectrum/(d*d*d);
	d = length(pos*vec2(.5,50.))*SCALE;
	col += spectrum/(d*d*d);

	return col;
}



mat2 Rot(float a) {
    float s=sin(a), c=cos(a);
    return mat2(c, -s, s, c);
}



float Star(vec2 uv, float flare) {
	float d = length(uv);
    float m = .05/d;
    
    float rays = max(0., 1.-abs(uv.x*uv.y*1000.));
    m += rays*flare;
    uv *= Rot(3.1415/4.);
    rays = max(0., 1.-abs(uv.x*uv.y*1000.));
    m += rays*.3*flare;
    
    m *= smoothstep(1., .2, d);
    return m;
}

float Hash21(vec2 p) {
    p = fract(p*vec2(123.34, 456.21));
    p += dot(p, p+45.32);
    return fract(p.x*p.y);
}

const float scale = 12.0;

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    //vec4 starColor = stars(vec4(1.0, 1.0, 1.0, 1.0), fragCoord);

    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec4 color = texture( iChannel0, uv );
    

    
    vec3 col = vec3(0);
	
    uv *= scale;
    
    vec2 gv = fract(uv)-.5;
    vec2 id = floor(uv);
    
    for(int y=-1;y<=1;y++) {
    	for(int x=-1;x<=1;x++) {
            vec2 offs = vec2(x, y);
            
    		float n = Hash21(id+offs); // random between 0 and 1
            float size = fract(n*345.32);
            
            col += Star(gv-offs-vec2(n, fract(n*34.))+.5, 1.);
        }
    }
    
    
    
    
    fragColor =  vec4(col, 1.0);

}
