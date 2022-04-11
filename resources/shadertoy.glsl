

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
        O += flare (U) * 0.01;           // rotating flare at random location
              //* r1(vec4(i+.2))                          // random scale
              //* (1.+sin(iDate.w+r1(vec4(i+.3))*6.))*.1  // time pulse
            //* (1.+.1*sr3(i+.4));               // random color - uncorrelated
              //* (1.+.1*sr3(i));                  // random color - correlated
    return O;
}

const float scale = 20.0;

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec4 starColor = stars(vec4(1.0, 1.0, 1.0, 1.0), fragCoord);

    vec2 uv = fragCoord/iResolution.xy;
    vec4 color = texture( iChannel0, uv );
    
    vec2 scaledUV = uv * scale;
    vec2 cell = floor(scaledUV);
    vec2 offset = scaledUV - cell;
    
    for(int i = -1; i<= 0; i++)
    {
        for(int j = -1; j<= 0; j++)
        {
            vec2 cell_t = cell + vec2(float(i), float(j));
            vec2 offset_t = offset - vec2(float(i), float(j));
            
            offset_t -= vec2(0.5, 0.5);
            
            float radius2 = dot(offset_t, offset_t);
            
            if(radius2 < 0.25)
            {
                fragColor = vec4(1.0, 1.0, 1.0, 1.0) * starColor;
            }
            else
                fragColor = color;
        }
    }
    
    
	//fragColor = color * starColor;
}
