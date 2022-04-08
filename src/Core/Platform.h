#ifndef _PLATFORM_H_
#define _PLATFORM_H_

#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace gl
{
	enum struct DataType
	{
		BYTE = GL_BYTE,
		UNSIGNED_BYTE = GL_UNSIGNED_BYTE,
		SHORT = GL_SHORT,
		UNSIGNED_SHORT = GL_UNSIGNED_SHORT,
		INT = GL_INT,
		UNSIGNED_INT = GL_UNSIGNED_INT,
		HALF_FLOAT = GL_HALF_FLOAT,
		FLOAT = GL_FLOAT,
		DOUBLE = GL_DOUBLE
	};
}



#define ASSERT(x) if(!(x)) __debugbreak();

#endif