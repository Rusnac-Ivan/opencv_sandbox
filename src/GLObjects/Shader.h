#ifndef _GL_SHADER_H_
#define _GL_SHADER_H_

#include <Core/Platform.h>

namespace gl
{
    enum struct ShaderType
    {
        VERTEX = GL_VERTEX_SHADER,
        FRAGMENT = GL_FRAGMENT_SHADER,
        GEOMETRY = GL_GEOMETRY_SHADER,
        TESS_CONTROL = GL_TESS_CONTROL_SHADER,
        TESS_EVALUATION = GL_TESS_EVALUATION_SHADER
    };

    class IShader
    {
    public:
        IShader() {}
        ~IShader() {}

        virtual void LoadSources(const unsigned short &count, const char **shaderCodes, const int *codeLengths) = 0;
        virtual ShaderType GetType() = 0;
        virtual unsigned int GetID() = 0;
    };

    template <ShaderType Shader_Type>
    class Shader : public IShader
    {
    private:
        unsigned int mID;

    public:
        Shader();
        ~Shader();

        void LoadSources(const unsigned short& count, const char** shaderCodes, const int* codeLengths);

        ShaderType GetType() { return Shader_Type; }
        unsigned int GetID() { return mID; }

    private:
        void CheckCompile();
    };

}

#endif
