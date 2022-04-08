#include "Program.h"
#include <glad/glad.h>
#include <stdarg.h>
#include <stdio.h>
#include <cstdlib>
#include <cassert>
#include "Shader.h"

namespace gl
{
    Program::Program() : mID(0)
    {
        mID = glCreateProgram();
        assert(mID != 0 && "Failed to create program!");
    }

    Program::~Program()
    {
        if (mID)
        {
            glDeleteProgram(mID);
            mID = 0;
        }
    }

    void Program::Attach(IShader *shader, ...)
    {
        va_list vl;
        va_start(vl, shader);
        do
        {
            glAttachShader(mID, shader->GetID());
        } while ((shader = va_arg(vl, IShader *)) != NULL);
        va_end(vl);
    }

    void Program::Use()
    {
        glUseProgram(mID);
    }

    inline bool Program::IsInUse()
    {
        int programId;
        glGetIntegerv(GL_CURRENT_PROGRAM, &programId);
        return mID == static_cast<uint32_t>(programId);
    }

    void Program::StopUsing()
    {
        glUseProgram(0);
    }

    void Program::Link()
    {
        glLinkProgram(mID);
        CheckLinkProgramErrors();
    }

    unsigned int Program::Uniform(const char *name) const
    {
        int32_t location = glGetUniformLocation(mID, reinterpret_cast<const GLchar *>(name));
        assert(location >= 0 && "Invalid uniform name!");
        return location;
    }

    int Program::GetAttribLocation(const char *name) const
    {
        int32_t attribLocation = glGetAttribLocation(mID, reinterpret_cast<const GLchar *>(name));
        assert(attribLocation >= 0 && "Invalid attribute name!");
        return attribLocation;
    }

    void Program::SetBool(const unsigned short &location, const bool &value) const
    {
        glUniform1i(location, (int)value);
    }

    void Program::SetInt(const unsigned short &location, const int &value) const
    {
        glUniform1i(location, value);
    }

    void Program::SetFloat(const unsigned short &location, const float &value) const
    {
        glUniform1f(location, value);
    }

    void Program::SetFloat2(const unsigned short &location, const float &x, const float &y) const
    {
        glUniform2f(location, x, y);
    }

    void Program::SetFloat2(const unsigned short &location, const glm::vec2 &value) const
    {
        glUniform2fv(location, 1, &value[0]);
    }

    void Program::SetFloat3(const unsigned short &location, const float &x, const float &y, const float &z) const
    {
        glUniform3f(location, x, y, z);
    }

    void Program::SetFloat3(const unsigned short &location, const glm::vec3 &value) const
    {
        glUniform3fv(location, 1, &value[0]);
    }

    void Program::SetFloat4(const unsigned short &location, const float &x, const float &y, const float &z, const float &w) const
    {
        glUniform4f(location, x, y, z, w);
    }

    void Program::SetFloat4(const unsigned short &location, const glm::vec4 &value) const
    {
        glUniform4fv(location, 1, &value[0]);
    }

    void Program::SetMatrix2(const unsigned short &location, const glm::mat2 &mat) const
    {
        glUniformMatrix2fv(location, 1, GL_FALSE, &mat[0][0]);
    }

    void Program::SetMatrix3(const unsigned short &location, const glm::mat3 &mat) const
    {
        glUniformMatrix3fv(location, 1, GL_FALSE, &mat[0][0]);
    }

    void Program::SetMatrix4(const unsigned short &location, const glm::mat4 &mat) const
    {

        glUniformMatrix4fv(location, 1, GL_FALSE, &mat[0][0]);
    }

    void Program::CheckLinkProgramErrors()
    {
        GLint success;
        char infoLog[1024];

        glGetProgramiv(mID, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(mID, 1024, NULL, infoLog);
            fprintf(stderr, "Failed to compile shader: %s\n", infoLog);
            exit(1);
        }
    }
}