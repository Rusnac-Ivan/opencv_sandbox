#include "RubiksCube.h"
#include <GLObjects/Program.h>
#include <glm/gtc/matrix_transform.hpp>

static float cube[] = {
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
    -0.3f, 0.3f, -0.3f, 0.0f, 1.0f, 0.0f
};

RubiksCube::RubiksCube()
{
    /*mModels.resize(3);

    for (std::vector< std::vector<glm::mat4>>& layer : mModels)
    {
        layer.resize(3);

        for (std::vector< glm::mat4>& cube : layer)
        {
            cube.resize(3);
        }
    }*/
    mCube.AddVBO(std::vector<gl::AttribType>({ gl::AttribType::POSITION, gl::AttribType::NORMAL }), 36, sizeof(cube), cube);

    GenerateModels();
}

RubiksCube::~RubiksCube()
{

}

void RubiksCube::GenerateModels()
{
    for (uint8_t i = 0; i < SIZE; i++)
    {
        for (uint8_t j = 0; j < SIZE; j++)
        {
            for (uint8_t k = 0; k < SIZE; k++)
            {
                mModels[i][j][k] = glm::mat4(1.f);
                glm::vec3 translate;
                translate.x = 0.5f * (i - 1);
                translate.y = 0.5f * (j - 1);
                translate.z = 0.5f * (k - 1);
                mModels[i][j][k] = glm::translate(mModels[i][j][k], translate);
            }
        }
    }
}

void RubiksCube::Draw(gl::Program* program)
{
    for (uint8_t i = 0; i < SIZE; i++)
    {
        for (uint8_t j = 0; j < SIZE; j++)
        {
            for (uint8_t k = 0; k < SIZE; k++)
            {
                
                program->SetMatrix4(program->Uniform("model"), mModels[i][j][k]);

                mCube.Draw(gl::Primitive::TRIANGLES);
            }
        }
    }
}