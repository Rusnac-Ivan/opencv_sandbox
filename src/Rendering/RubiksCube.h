#ifndef _RUBIKS_CUBE_H_
#define _RUBIKS_CUBE_H_

#include <GLObjects/Vertices.h>
//#include <GLObjects/Program.h>
#include <glm/glm.hpp>
#include <vector>

#define SIZE 3
namespace gl
{
    class Program;
}


class RubiksCube
{
    //std::vector<std::vector<std::vector<glm::mat4>>> mModels;

    glm::mat4 mModels[SIZE][SIZE][SIZE];

    gl::Vertices mCube;
    //gl::Program mProgram;

public:
	RubiksCube();
	~RubiksCube();

    void Draw(gl::Program* program);
private:
    void GenerateModels();
};



#endif