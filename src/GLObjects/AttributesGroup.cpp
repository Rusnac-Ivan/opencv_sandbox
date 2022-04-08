#include "AttributesGroup.h"
#include "VertexArray.h"
#include <cassert>

namespace gl
{
	static unsigned short DataSize(const DataType& dataType)
	{
		switch (dataType)
		{
			case DataType::BYTE:
				return sizeof(char);
			case DataType::UNSIGNED_BYTE:
				return sizeof(unsigned char);
			case DataType::SHORT:
				return sizeof(short);
			case DataType::UNSIGNED_SHORT:
				return sizeof(unsigned short);
			case DataType::INT:
				return sizeof(int);
			case DataType::UNSIGNED_INT:
				return sizeof(unsigned int);
			case DataType::FLOAT:
				return sizeof(float);
			case DataType::DOUBLE:
				return sizeof(double);
			default:
			{
				assert(false && "Invalid Data Type!");
				return -1;
			}
		}
	}

	static unsigned short Offset(const unsigned short &shift, const DataType &dataType)
	{
		return shift * DataSize(dataType);
	}

	AttributesGroup::AttributesGroup() : mIsFinished(false), mStrideSize(0)
	{}

	void AttributesGroup::PushAttribute(const unsigned short &layoutLocation, const DataType &dataType, const unsigned short &numOfComponents, const char *attributeName)
	{
		assert(mIsFinished == false && "Group is in finished state!");
		mAttributes.push_back({layoutLocation, dataType, numOfComponents, attributeName});
		mStrideSize += numOfComponents * DataSize(dataType);
	}

	void AttributesGroup::PopAttribute()
	{
		assert(mIsFinished == false && "Group is in finished state!");
		mStrideSize -= mAttributes.back().numOfComponents * DataSize(mAttributes.back().dataType);
		mAttributes.pop_back();
	}

	void AttributesGroup::Compose(VertexArray *vertArray, const unsigned int &dataSize, const void *data)
	{
		assert(mIsFinished == false && "Group is in finished state!");

		vertArray->Bind();
		mVBO.Bind(Buffer::Target::ARRAY);

		mVBO.Data(dataSize, data, Buffer::Access::STATIC_DRAW);
		

		unsigned short shift = 0;
		for (Attribute &attrib : mAttributes)
		{
			int offset = Offset(shift, attrib.dataType);
			vertArray->SetAttributePointer(attrib.layoutLocation, attrib.numOfComponents, attrib.dataType, false, mStrideSize, (const void *)(offset));
			vertArray->EnableAttribute(attrib.layoutLocation);
			shift += attrib.numOfComponents;
		}

		mVBO.UnBind();
		vertArray->UnBind();

		mIsFinished = true;
	}

	void AttributesGroup::Decompose(VertexArray *vertArray)
	{
		assert(mIsFinished == false && "Group is in unfinished state!");
		for (Attribute &attrib : mAttributes)
		{
			vertArray->DisableAttribute(attrib.layoutLocation);
		}

		mIsFinished = false;
	}
}
