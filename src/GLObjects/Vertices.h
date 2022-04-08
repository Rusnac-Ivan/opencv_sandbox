#ifndef _GL_VERTICES_H_
#define _GL_VERTICES_H_

#include <vector>
#include <Core/Platform.h>
#include "VertexArray.h"



namespace gl
{
    enum struct Primitive
    {
        POINTS = GL_POINTS,
        LINE_STRIP = GL_LINE_STRIP,
        LINE_LOOP = GL_LINE_LOOP,
        LINES = GL_LINES,
        TRIANGLE_STRIP = GL_TRIANGLE_STRIP, 
        TRIANGLE_FAN = GL_TRIANGLE_FAN,
        TRIANGLES = GL_TRIANGLES
    };

    enum struct AttribType
    {
        POSITION,
        NORMAL,
        TEXTURE_UV,
        COLOR,
        TANGENT,
        BITANGENT
    };

    class AttributesGroup;
    class Buffer;

    class Vertices
    {
    private:
        VertexArray mVertexArray;
        std::vector<AttributesGroup *> mAttributes;
        Buffer *mEBO;
        unsigned short mLayoutLocation;
        unsigned int mVertexCount;

    public:
        Vertices();
        ~Vertices();

        void AddVBO(std::vector<AttribType> &attribs, const unsigned int& vertexCount, const unsigned int &size, const void *data);
        void SetEBO(const unsigned int &size, const void *data);

        void Draw(const Primitive& mode);
        void Draw(const Primitive& mode, const unsigned int& instanceCount);

    private:
        void MakeAttribute(const AttribType &type, AttributesGroup *attribGroup);
    };
}

#endif