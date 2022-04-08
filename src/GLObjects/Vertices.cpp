#include "Vertices.h"
#include "AttributesGroup.h"
#include "Buffer.h"
#include <cassert>

namespace gl
{
    Vertices::Vertices()
    {
        mLayoutLocation = 0;
        mEBO = nullptr;
    }

    Vertices::~Vertices()
    {
        for (AttributesGroup* attrib : mAttributes)
        {
            delete attrib;
        }
        mAttributes.clear();
        if (mEBO)
            delete mEBO;
    }

    void Vertices::AddVBO(std::vector<AttribType>& attribTypes, const unsigned int& vertexCount, const unsigned int& dataSize, const void* data)
    {
        assert(attribTypes.size() > 0 && "No attribute types!");
        assert(dataSize > 0 && "No data size!");
        assert(data && "No data, is null!");
        mVertexCount = vertexCount;
        AttributesGroup* attribGroup = new AttributesGroup;
        for (const AttribType& type : attribTypes)
        {
            MakeAttribute(type, attribGroup);
        }
        attribGroup->Compose(&mVertexArray, dataSize, data);
        mAttributes.push_back(attribGroup);

        mLayoutLocation = 0;
    }

    void Vertices::SetEBO(const unsigned int& size, const void* data)
    {
        if (mEBO)
            delete mEBO;
        mEBO = new Buffer;
        //mEBO->SetCurrentBindingTarget(Buffer::Target::ELEMENT_ARRAY);
        //mEBO->SetCurrentAccessFrequency(Buffer::Access::STATIC_DRAW);
        mVertexArray.Bind();
        mEBO->Bind(Buffer::Target::ELEMENT_ARRAY);
        mEBO->Data(size, data, Buffer::Access::STATIC_DRAW);
        mEBO->UnBind();
        mVertexArray.UnBind();
    }

    void Vertices::MakeAttribute(const AttribType& type, AttributesGroup* attribGroup)
    {
        assert(attribGroup && "AttributesGroup is nullptr!");

        switch (type)
        {
            case AttribType::POSITION:
            {
                attribGroup->PushAttribute(mLayoutLocation++, DataType::FLOAT, 3, "aPos");
            }break;
            case AttribType::NORMAL:
            {
                attribGroup->PushAttribute(mLayoutLocation++, DataType::FLOAT, 3, "aNorm");
            }break;
            case AttribType::TEXTURE_UV:
            {
                attribGroup->PushAttribute(mLayoutLocation++, DataType::FLOAT, 2, "aUV");
            }break;
            case AttribType::COLOR:
            {
                attribGroup->PushAttribute(mLayoutLocation++, DataType::FLOAT, 3, "aCol");
            }break;
            case AttribType::TANGENT:
            {
                attribGroup->PushAttribute(mLayoutLocation++, DataType::FLOAT, 3, "aTan");
            }break;
            case AttribType::BITANGENT:
            {
                attribGroup->PushAttribute(mLayoutLocation++, DataType::FLOAT, 3, "aBitan");
            }break;
            default:
            {
                assert(false && "Invalid attribute type!");
            }break;
        }
    }

    void Vertices::Draw(const Primitive& mode)
    {
        mVertexArray.Bind();
        if (!mEBO)
        {
            glDrawArrays(static_cast<GLenum>(mode), 0, mVertexCount);
        }
        else
        {
            glDrawElements(static_cast<GLenum>(mode), mVertexCount, GL_UNSIGNED_BYTE, nullptr);
        }
        mVertexArray.UnBind();
    }
    void Vertices::Draw(const Primitive& mode, const unsigned int& instanceCount)
    {
        mVertexArray.Bind();
        if (!mEBO)
        {
            glDrawArraysInstanced(static_cast<GLenum>(mode), 0, mVertexCount, instanceCount);
        }
        else
        {
            glDrawElementsInstanced(static_cast<GLenum>(mode), mVertexCount, GL_UNSIGNED_BYTE, nullptr, instanceCount);
        }
        mVertexArray.UnBind();
    }
}

