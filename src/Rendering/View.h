#ifndef _VIEW_H_
#define _VIEW_H_

#include <GLObjects/Vertices.h>
#include <GLObjects/RenderContext.h>
//#include <Tools/Camera.h>
//#include <Tools/Frustum.h>
#include <GLObjects/Program.h>
#include <Rendering/RubiksCube.h>

class View
{
private:
    //Camera mCamera;
    //Perspective mPerspective;
    
    gl::Vertices mBishop;
    gl::Vertices mKnight;

    gl::Program mProgram;
    float mWidth;
    float mHeight;
    //RubiksCube mRubiksCube;

public:
    View();
    ~View();
    

    void OnInitialize();
    void OnUpdate();
    void OnFinalize();

    void OnMouseLeftDown(double x, double y);
    void OnMouseLeftUp(double x, double y);
    void OnMouseRightDown(double x, double y);
    void OnMouseRightUp(double x, double y);
    void OnMouseMiddleDown(double x, double y);
    void OnMouseMiddleUp(double x, double y);
    void OnMouseMove(double x, double y);
    void OnMouseWhell(double offset);

    void OnKeyPress(int key);
    void OnKeyRepeat(int key);
    void OnKeyRelease(int key);

    void OnResize(int width, int height);

    void OnCreate(int width, int height);
    void OnDestroy();
private:
    
};

#endif