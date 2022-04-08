#ifndef _GL_VERTEX_ARRAY_H_
#define _GL_VERTEX_ARRAY_H_

#include <Core/Platform.h>

namespace gl
{

	

	class VertexArray
	{
	private:
		unsigned int mID;

	public:
		VertexArray();
		~VertexArray();

		void Bind();
		void UnBind();

		void SetAttributePointer(const unsigned short &layoutLocation, const unsigned short &attribSize, const DataType &dtaType, const bool &normalize, const unsigned short &strideSize, const void *pointer);
		void EnableAttribute(const unsigned short &layoutLocation);
		void DisableAttribute(const unsigned short &layoutLocation);
	};
}

#endif