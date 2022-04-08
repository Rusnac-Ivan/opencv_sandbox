#include "View.h"
#include <GLObjects/Pipeline.h>
#include <GLObjects/Shader.h>
#include <vector>
#include <fstream>

#define GLSL(str) (const char *)"#version 330\n" #str

float vertices[] = {
    -0.3f, -0.3f, -0.3f, 0.0f, 0.0f, -1.0f,
    0.3f, -0.3f, -0.3f, 0.0f, 0.0f, -1.0f,
    0.3f, 0.3f, -0.3f, 0.0f, 0.0f, -1.0f,
    0.3f, 0.3f, -0.3f, 0.0f, 0.0f, -1.0f,
    -0.3f, 0.3f, -0.3f, 0.0f, 0.0f, -1.0f,
    -0.3f, -0.3f, -0.3f, 0.0f, 0.0f, -1.0f,

    -0.3f, -0.3f, 0.3f, 0.0f, 0.0f, 1.0f,
    0.3f, -0.3f, 0.3f, 0.0f, 0.0f, 1.0f,
    0.3f, 0.3f, 0.3f, 0.0f, 0.0f, 1.0f,
    0.3f, 0.3f, 0.3f, 0.0f, 0.0f, 1.0f,
    -0.3f, 0.3f, 0.3f, 0.0f, 0.0f, 1.0f,
    -0.3f, -0.3f, 0.3f, 0.0f, 0.0f, 1.0f,

    -0.3f, 0.3f, 0.3f, -1.0f, 0.0f, 0.0f,
    -0.3f, 0.3f, -0.3f, -1.0f, 0.0f, 0.0f,
    -0.3f, -0.3f, -0.3f, -1.0f, 0.0f, 0.0f,
    -0.3f, -0.3f, -0.3f, -1.0f, 0.0f, 0.0f,
    -0.3f, -0.3f, 0.3f, -1.0f, 0.0f, 0.0f,
    -0.3f, 0.3f, 0.3f, -1.0f, 0.0f, 0.0f,

    0.3f, 0.3f, 0.3f, 1.0f, 0.0f, 0.0f,
    0.3f, 0.3f, -0.3f, 1.0f, 0.0f, 0.0f,
    0.3f, -0.3f, -0.3f, 1.0f, 0.0f, 0.0f,
    0.3f, -0.3f, -0.3f, 1.0f, 0.0f, 0.0f,
    0.3f, -0.3f, 0.3f, 1.0f, 0.0f, 0.0f,
    0.3f, 0.3f, 0.3f, 1.0f, 0.0f, 0.0f,

    -0.3f, -0.3f, -0.3f, 0.0f, -1.0f, 0.0f,
    0.3f, -0.3f, -0.3f, 0.0f, -1.0f, 0.0f,
    0.3f, -0.3f, 0.3f, 0.0f, -1.0f, 0.0f,
    0.3f, -0.3f, 0.3f, 0.0f, -1.0f, 0.0f,
    -0.3f, -0.3f, 0.3f, 0.0f, -1.0f, 0.0f,
    -0.3f, -0.3f, -0.3f, 0.0f, -1.0f, 0.0f,

    -0.3f, 0.3f, -0.3f, 0.0f, 1.0f, 0.0f,
    0.3f, 0.3f, -0.3f, 0.0f, 1.0f, 0.0f,
    0.3f, 0.3f, 0.3f, 0.0f, 1.0f, 0.0f,
    0.3f, 0.3f, 0.3f, 0.0f, 1.0f, 0.0f,
    -0.3f, 0.3f, 0.3f, 0.0f, 1.0f, 0.0f,
    -0.3f, 0.3f, -0.3f, 0.0f, 1.0f, 0.0f};

View::View()
{
}
View::~View()
{
}

void View::OnCreate(int width, int height)
{
    mWidth = width;
    mHeight = height;
}
void View::OnDestroy()
{
}

void View::OnInitialize()
{
    gl::RenderContext::SetClearColor(0.0f, 0.3f, 0.2f, 1.00f);
    //gl::Pipeline::EnableDepthTest();

    //mBishop.AddVBO(std::vector<gl::AttribType>({gl::AttribType::POSITION, gl::AttribType::NORMAL}), __bishop_vert_count, sizeof(__bishop_vert), __bishop_vert);
    //mKnight.AddVBO(std::vector<gl::AttribType>({gl::AttribType::POSITION, gl::AttribType::NORMAL}), __knight_vert_count, sizeof(__knight_vert), __knight_vert);

    /*const char* vertShader = GLSL(
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aNorm;

        float rand(float x) {
            return fract(sin(x) * 100000.0);
        }

        out vec3 FragPos;
        out vec3 Normal;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main()
        {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = aNorm;

            gl_Position = projection * view * vec4(FragPos, 1.0);
        });
    int vertShSize = strlen(vertShader);
    const char *fragShader = GLSL(
        out vec4 FragColor;

        in vec3 Normal;
        in vec3 FragPos;

        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec3 lightColor;
        uniform vec3 objectColor;

        void main()
        {
            // ambient
            float ambientStrength = 0.1;
            vec3 ambient = ambientStrength * lightColor;

            // diffuse
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            // specular
            float specularStrength = 0.5;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * lightColor;

            vec3 result = (ambient + diffuse + specular) * objectColor;
            FragColor = vec4(result, 1.0);
        });
    int fragShSize = strlen(fragShader);

    gl::Shader<gl::ShaderType::VERTEX> vertSh;
    gl::Shader<gl::ShaderType::FRAGMENT> fragSh;

    vertSh.LoadSources(1, &vertShader, &vertShSize);
    fragSh.LoadSources(1, &fragShader, &fragShSize);

    mProgram.Attach(&vertSh, &fragSh, NULL);

    mProgram.Link();

    glm::mat4 model(1.f);

    //mPerspective.SetRange(0.01f, 100.f);
    //mPerspective.SetFieldOfView(60.f);
    //mPerspective.ComputeAspectRatio(mWidth, mHeight);

    mCamera.Init(glm::vec2(mWidth, mHeight), glm::vec3(0.f, 0.f, -2.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 1.f, 0.f));

    mProgram.Use();
    mProgram.SetMatrix4(mProgram.Uniform("model"), model);
    mProgram.SetMatrix4(mProgram.Uniform("projection"), mCamera.GetProjectMat());
    mProgram.SetFloat3(mProgram.Uniform("lightColor"), glm::vec3(1.f, 1.f, 1.f));
    mProgram.SetFloat3(mProgram.Uniform("objectColor"), glm::vec3(0.714f, 0.4284, 0.18144));
    mProgram.StopUsing();*/
}

void View::OnUpdate()
{
    //mProgram.Use();

    //mProgram.SetFloat3(mProgram.Uniform("lightPos"), mCamera.GetPosition());
    //mProgram.SetFloat3(mProgram.Uniform("viewPos"), mCamera.GetPosition());

    //glm::mat4 model = glm::mat4(1.0f);

    //glm::mat4 model1 = glm::translate(model, glm::vec3(-0.4f, 0.f, 0.f));
    //glm::mat4 model2 = glm::translate(model, glm::vec3(0.4f, 0.f, 0.f));

    //glm::mat4 model1 = glm::scale(model, glm::vec3(1.f, 5.f, 5.f));

    //mProgram.SetMatrix4(mProgram.Uniform("view"), mCamera.GetViewMat());

    gl::RenderContext::Clear(gl::BufferBit::COLOR, gl::BufferBit::DEPTH);

    //mProgram.SetMatrix4(mProgram.Uniform("model"), model1);
    //mBishop.Draw(gl::Primitive::TRIANGLES);
    //mProgram.SetMatrix4(mProgram.Uniform("model"), model2);
    //mKnight.Draw(gl::Primitive::TRIANGLES);

    //mProgram.StopUsing();

    //mCamera.Update();
}
void View::OnFinalize()
{
}

void View::OnMouseLeftDown(double x, double y)
{
    //mCamera.BeginDrag((float)x, (float)y);
}
void View::OnMouseLeftUp(double x, double y)
{
    //mCamera.EndDrag();
}
void View::OnMouseRightDown(double x, double y)
{
    //mCamera.BeginPitch((float)x, (float)y);
}
void View::OnMouseRightUp(double x, double y)
{
    //mCamera.EndPitch();
}
void View::OnMouseMiddleDown(double x, double y)
{
}
void View::OnMouseMiddleUp(double x, double y)
{
}
void View::OnMouseMove(double x, double y)
{

    //mCamera.Drag((float)x, (float)y);
    //mCamera.Pitch((float)x, (float)y);
}
void View::OnMouseWhell(double offset)
{
    //mCamera.Wheel(offset);
}

void View::OnKeyPress(int key)
{
}
void View::OnKeyRepeat(int key)
{
}
void View::OnKeyRelease(int key)
{
}

void View::OnResize(int width, int height)
{
    mWidth = width;
    mHeight = height;
    gl::RenderContext::SetViewport(width, height);
    /*mCamera.Resize(glm::vec2(width, height));
    mProgram.Use();
    mProgram.SetMatrix4(mProgram.Uniform("projection"), mCamera.GetProjectMat());
    mProgram.StopUsing();*/
}