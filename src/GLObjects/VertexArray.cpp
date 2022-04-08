#include "VertexArray.h"
#include <cassert>

namespace gl
{
	VertexArray::VertexArray() : mID(0)
	{
		glGenVertexArrays(1, &mID);
	}

	VertexArray::~VertexArray()
	{
		if (!mID)
		{
			glDeleteVertexArrays(1, &mID);
			mID = 0;
		}
	}

	void VertexArray::Bind()
	{
		glBindVertexArray(mID);
	}

	void VertexArray::UnBind()
	{
		glBindVertexArray(0);
	}

	void VertexArray::SetAttributePointer(const unsigned short &layoutLocation, const unsigned short &attribSize, const DataType &dtaType, const bool &normalize, const unsigned short &strideSize, const void *pointer)
	{
#ifndef NDEBUG
		int currentBinding;
		glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &currentBinding);
		assert(currentBinding == mID && "Not conformity Vertex Array Object!");
#endif

		glVertexAttribPointer(layoutLocation, attribSize, static_cast<GLenum>(dtaType), normalize, strideSize, (const GLvoid*)pointer);
	}

	void VertexArray::EnableAttribute(const unsigned short &layoutLocation)
	{
#ifndef NDEBUG
		int currentBinding;
		glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &currentBinding);
		assert(currentBinding == mID && "Not conformity Vertex Array Object!");
#endif

		glEnableVertexAttribArray(layoutLocation);
	}

	void VertexArray::DisableAttribute(const unsigned short &layoutLocation)
	{
#ifndef NDEBUG
		int currentBinding;
		glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &currentBinding);
		assert(currentBinding == mID && "Not conformity Vertex Array Object!");
#endif

		glDisableVertexAttribArray(layoutLocation);
	}
}
