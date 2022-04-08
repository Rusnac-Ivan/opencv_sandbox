#include "Shader.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>

namespace gl
{
	template <ShaderType Shader_Type>
	Shader<Shader_Type>::Shader() : IShader(), mID(0)
	{
		mID = glCreateShader(static_cast<GLenum>(Shader_Type));
		assert(mID != 0 && "Failed to create shader!");
	}

	template <ShaderType Shader_Type>
	Shader<Shader_Type>::~Shader()
	{
		if (!mID)
		{
			glDeleteShader(mID);
		}
	}

	template <ShaderType Shader_Type>
	void Shader<Shader_Type>::LoadSources(const unsigned short &count, const char **shaderCodes, const int *codeLengths)
	{
		glShaderSource(mID, count, shaderCodes, codeLengths);

		glCompileShader(mID);

		CheckCompile();
	}

	template <ShaderType Shader_Type>
	void Shader<Shader_Type>::CheckCompile()
	{
		GLint is_success;
		char infoLog[1024];

		glGetShaderiv(mID, GL_COMPILE_STATUS, &is_success);
		if (!is_success)
		{

			glGetShaderInfoLog(mID, 1024, nullptr, infoLog);
			fprintf(stderr, "Failed to compile shader: %s\n", infoLog);

			glDeleteShader(mID);
			mID = 0;

			exit(1);
		}
	}

	template class Shader<ShaderType::VERTEX>;
	template class Shader<ShaderType::FRAGMENT>;
	template class Shader<ShaderType::GEOMETRY>;
	template class Shader<ShaderType::TESS_CONTROL>;
	template class Shader<ShaderType::TESS_EVALUATION>;
}