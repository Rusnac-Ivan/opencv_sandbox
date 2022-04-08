#ifndef _GL_BUFFER_H_
#define _GL_BUFFER_H_

#include <Core/Platform.h>

namespace gl
{
	class Buffer
	{
		public:
			enum struct Target
			{
				UNKNOWN				= GL_FALSE,
				ARRAY				= GL_ARRAY_BUFFER,
				COPY_READ			= GL_COPY_READ_BUFFER,
				COPY_WRITE			= GL_COPY_WRITE_BUFFER,
				DRAW_INDIRECT		= GL_DRAW_INDIRECT_BUFFER,
				ELEMENT_ARRAY		= GL_ELEMENT_ARRAY_BUFFER,
				PIXEL_PACK			= GL_PIXEL_PACK_BUFFER,
				PIXEL_UNPACK		= GL_PIXEL_UNPACK_BUFFER,
				TEXTURE				= GL_TEXTURE_BUFFER,
				TRANSFORM_FEEDBACK	= GL_TRANSFORM_FEEDBACK_BUFFER,
				UNIFORM				= GL_UNIFORM_BUFFER
			};
			enum struct Access
			{
				UNKNOWN			= GL_FALSE,
				
				STREAM_DRAW		= GL_STREAM_DRAW,
				STREAM_READ		= GL_STREAM_READ,
				STREAM_COPY		= GL_STREAM_COPY,
				
				STATIC_DRAW		= GL_STATIC_DRAW,
				STATIC_READ		= GL_STATIC_READ,
				STATIC_COPY		= GL_STATIC_COPY,
				
				DYNAMIC_DRAW	= GL_DYNAMIC_DRAW,
				DYNAMIC_READ	= GL_DYNAMIC_READ,
				DYNAMIC_COPY	= GL_DYNAMIC_COPY
			};
		private:
			unsigned int	mID;
			Target	mCurrentTarget;
			Access	mAccessFreq;
		public:
			Buffer();
			~Buffer();

			Buffer(const Buffer&) = delete;
			Buffer& operator=(const Buffer&) = delete;

			Buffer(Buffer&&) noexcept;
			Buffer& operator=(Buffer&&) noexcept;

			void Bind(Target target);
			void UnBind();

			const Target& GetCurrentTarget() const { return mCurrentTarget; }
			const Access& GetCurrentAccessFrequency() const { return mAccessFreq; }

			//load or update data
			//void LoadData(const unsigned int& size, const void* data) const;
			//void Update(const unsigned short& offset, const unsigned short& size, const void* data) const;

			void Data(const unsigned int& size, const void* data, Access accessFreq);
			void SubData(const unsigned short& offset, const unsigned short& size, const void* data);
			
	};
}

#endif
