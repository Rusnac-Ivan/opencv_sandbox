#ifndef _GL_ATTRIBUTES_GROUP_H_
#define _GL_ATTRIBUTES_GROUP_H_

#include <list>
#include <string>
#include "Buffer.h"


namespace gl
{
	class VertexArray;

	struct Attribute
	{
		unsigned short layoutLocation;
		DataType dataType;
		unsigned short numOfComponents;
		std::string layoutName;
	};

	class AttributesGroup
	{
		Buffer mVBO;
		bool mIsFinished;
		unsigned short mStrideSize;
		std::list<Attribute> mAttributes;

	public:
		AttributesGroup();
		~AttributesGroup() {}

		void PushAttribute(const unsigned short &layoutLocation, const DataType &dataType, const unsigned short &numOfComponents, const char *attributeName);
		void PopAttribute();

		void Compose(VertexArray *vertArray, const unsigned int &dataSize, const void *data);
		void Decompose(VertexArray *vertArray);
	};

}

#endif
