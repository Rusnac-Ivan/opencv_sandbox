#ifndef _GL_TEXTURE_
#define _GL_TEXTURE_

#include <Core/Platform.h>
#include <cassert>
#include <map>

namespace gl
{
	
	enum struct TextureType
	{
		TARGET_1D					= GL_TEXTURE_1D,
		TARGET_2D					= GL_TEXTURE_2D,
		TARGET_3D					= GL_TEXTURE_3D,
		TARGET_1D_ARRAY				= GL_TEXTURE_1D_ARRAY,
		TARGET_2D_ARRAY				= GL_TEXTURE_2D_ARRAY,
		TARGET_RECTANGLE			= GL_TEXTURE_RECTANGLE,
		
		TARGET_CUBE_MAP				= GL_TEXTURE_CUBE_MAP,
		TARGET_CUBE_MAP_ARRAY		= GL_TEXTURE_CUBE_MAP_ARRAY,
		TARGET_BUFFER				= GL_TEXTURE_BUFFER,
		TARGET_2D_MULTISAMPLE		= GL_TEXTURE_2D_MULTISAMPLE,
		TARGET_2D_MULTISAMPLE_ARRAY	= GL_TEXTURE_2D_MULTISAMPLE_ARRAY
	};

	

	enum struct Format
	{
		//Base
		RED = GL_RED,
		RG = GL_RG,
		RGB = GL_RGB,
		RGBA = GL_RGBA,
		DEPTH = GL_DEPTH_COMPONENT,
		DEPTH_STENCIL = GL_DEPTH_STENCIL,

		//Sized
		R8 = GL_R8,
		R16 = GL_R16,
		RG8 = GL_RG8,
		RG16 = GL_RG16,
		R3_G3_B2 = GL_R3_G3_B2,
		RGB4 = GL_RGB4,
		RGB5 = GL_RGB5,
		RGB8 = GL_RGB8,
		RGB10 = GL_RGB10,
		RGB12 = GL_RGB12,
		RGBA2 = GL_RGBA2,
		RGBA4 = GL_RGBA4,
		RGBA8 = GL_RGBA8,

		//Compressed
		COMPRESSED_RED = GL_COMPRESSED_RED,
		COMPRESSED_RG = GL_COMPRESSED_RG,
		COMPRESSED_RGB = GL_COMPRESSED_RGB,
		COMPRESSED_RGBA = GL_COMPRESSED_RGBA,
		COMPRESSED_SRGB = GL_COMPRESSED_SRGB,
		COMPRESSED_SRGB_ALPHA = GL_COMPRESSED_SRGB_ALPHA,
		COMPRESSED_RED_RGTC1 = GL_COMPRESSED_RED_RGTC1,
		COMPRESSED_SIGNED_RED_RGTC1 = GL_COMPRESSED_SIGNED_RED_RGTC1,

		UNKNOWN = 0xFFFFFFFF
	};
	
	enum struct ParamName
	{
		DEPTH_STENCIL_MODE = GL_DEPTH_STENCIL_TEXTURE_MODE, 
		BASE_LEVEL = GL_TEXTURE_BASE_LEVEL,
		COMPARE_FUNC = GL_TEXTURE_COMPARE_FUNC,
		COMPARE_MODE = GL_TEXTURE_COMPARE_MODE,
		LOD_BIAS = GL_TEXTURE_LOD_BIAS,
		MIN_FILTER = GL_TEXTURE_MIN_FILTER,
		MAG_FILTER = GL_TEXTURE_MAG_FILTER,
		MIN_LOD = GL_TEXTURE_MIN_LOD,
		MAX_LOD = GL_TEXTURE_MAX_LOD,
		MAX_LEVEL = GL_TEXTURE_MAX_LEVEL,
		SWIZZLE_G = GL_TEXTURE_SWIZZLE_G,
		SWIZZLE_B = GL_TEXTURE_SWIZZLE_B,
		SWIZZLE_A = GL_TEXTURE_SWIZZLE_A,
		WRAP_S = GL_TEXTURE_WRAP_S,
		WRAP_T = GL_TEXTURE_WRAP_T,
		WRAP_R = GL_TEXTURE_WRAP_R,

		NONE = 0xFFFFFFFF,
	};

	enum struct ParamValue
	{
		NEAREST = GL_NEAREST,
		LINEAR = GL_LINEAR,
		NEAREST_MIPMAP_NEAREST = GL_NEAREST_MIPMAP_NEAREST,
		NEAREST_MIPMAP_LINEAR = GL_NEAREST_MIPMAP_LINEAR,
		LINEAR_MIPMAP_NEAREST = GL_LINEAR_MIPMAP_NEAREST,
		LINEAR_MIPMAP_LINEAR = GL_LINEAR_MIPMAP_LINEAR,

		CLAMP_TO_EDGE = GL_CLAMP_TO_EDGE,
		CLAMP_TO_BORDER = GL_CLAMP_TO_BORDER,

		REPEAT = GL_REPEAT,

		MIRROR_CLAMP_TO_EDGE = GL_MIRROR_CLAMP_TO_EDGE,
		MIRRORED_REPEAT = GL_MIRRORED_REPEAT,

		UNKNOWN = 0xFFFFFFFF,
	};


	template<TextureType Target>
	class Texture
	{
		private:
			unsigned int mID;
		protected:
			const TextureType mTarget = Target;
		public:
			Texture();
			virtual ~Texture();

			void Bind();
			void UnBind();

			void Activate(const unsigned short& unit);
			void Deactivate(const unsigned short& unit);

			unsigned int GetId() { return mID; }
	};

	template <TextureType Target>
	Texture<Target>::Texture()
	{
		glGenTextures(1, &mID);
		assert(mID != 0 && "Failed to generate texture!");
	}

	template <TextureType Target>
	Texture<Target>::~Texture()
	{
		if (mID != 0)
			glDeleteTextures(1, &mID);
	}

	template <TextureType Target>
	void Texture<Target>::Bind()
	{
		GLenum target = static_cast<GLenum>(mTarget);
		glBindTexture(target, mID);
	}

	template <TextureType Target>
	void Texture<Target>::UnBind()
	{
		GLenum target = static_cast<GLenum>(mTarget);
		glBindTexture(target, NULL);
	}

	template <TextureType Target>
	void Texture<Target>::Activate(const unsigned short& unit)
	{
		GLenum target = static_cast<GLenum>(mTarget);
		glActiveTexture(GL_TEXTURE0 + unit);
		glBindTexture(target, mID);
	}

	template <TextureType Target>
	void Texture<Target>::Deactivate(const unsigned short& unit)
	{
		GLenum target = static_cast<GLenum>(mTarget);
		glActiveTexture(GL_TEXTURE0 + unit);
		glBindTexture(target, NULL);
	}


	class Texture2D : public Texture<TextureType::TARGET_2D>
	{
			uint32_t	mWidth;
			uint32_t	mHeight;
			Format		mFormat;
			Format		mInternalFormat;
			DataType	mPixelDataFormat;
			uint8_t*	mRawData;
			uint32_t	mRawDataSize;
			bool		mGenerateMipmap;

			std::map<ParamName, ParamValue> mParameters;
		public:
			Texture2D();
			virtual ~Texture2D();

			uint32_t GetWidth() { return mWidth; };
			uint32_t GetHeight() { return mHeight; };
			//ParamValue GetWrapS() { return mParameters[ParamName::WRAP_S]; };
			//ParamValue GetWrapT() { return mParameters[ParamName::WRAP_T]; };
			//ParamValue GetMagFilter() { return mParameters[ParamName::MAG_FILTER]; };
			//ParamValue GetMinFilter() { return mParameters[ParamName::MIN_FILTER]; };
			bool GetGenerateMipmap() { return mGenerateMipmap; };
			Format GetFormat() { return mFormat; };
			Format GetInternalFormat() { return mInternalFormat; };

			void SetParameters(const std::map<ParamName, ParamValue>& param_val);

			//void SetWidth(uint32_t width) { mWidth = width; };
			//void SetHeight(uint32_t height) { mHeight = height; };
			
			//void SetFormat(Format format) { mFormat = format; };
			//void SetInternalFormat(Format internalFormat) { mInternalFormat = internalFormat; };

			void SetTexture2D(int32_t level, Format internalformat, uint32_t width, uint32_t height, uint32_t border, Format format, DataType type, void* pixels);
			//void SetMipmapState(uint32_t generateMipmap);

			void Update(uint32_t x, uint32_t y, uint32_t width, uint32_t height, void* buffer);

		private:

	};

	Texture2D::Texture2D() : Texture<TextureType::TARGET_2D>(), 
		mWidth(NULL),
		mHeight(NULL),
		mFormat(Format::UNKNOWN),
		mInternalFormat(Format::UNKNOWN),
		mPixelDataFormat(DataType::UNSIGNED_BYTE),
		mRawData(nullptr),
		mRawDataSize(NULL),
		mGenerateMipmap(false)
	{
	}

	Texture2D::~Texture2D()
	{
	}

	void Texture2D::SetTexture2D(int32_t level, Format internalformat, uint32_t width, uint32_t height, uint32_t border, Format format, DataType type, void* pixels)
	{
		mWidth = width;
		mHeight = height;
		mInternalFormat = internalformat;
		mFormat = format;
		mPixelDataFormat = type;

		GLenum target = static_cast<GLenum>(mTarget);
		GLenum in_form = static_cast<GLenum>(internalformat);
		GLenum form = static_cast<GLenum>(format);
		GLenum data_type = static_cast<GLenum>(type);
		Bind();
		glTexImage2D(target, level, in_form, width, height, border, form, data_type, pixels);
		UnBind();
	}


	void Texture2D::SetParameters(const std::map<ParamName, ParamValue>& parameters)
	{
		GLenum target = static_cast<GLenum>(mTarget);
		Bind();
		for (auto& parameter_value : parameters)
		{
			mParameters[parameter_value.first] = parameter_value.second;

			GLenum param_name = static_cast<GLenum>(parameter_value.first);
			GLenum param_val = static_cast<GLenum>(parameter_value.second);

			glTexParameteri(target, param_name, param_val);
		}
		UnBind();
	}



	void Texture2D::Update(uint32_t x, uint32_t y, uint32_t width, uint32_t height, void* buffer)
	{
		GLenum target = static_cast<GLenum>(mTarget);
		GLenum in_form = static_cast<GLenum>(mInternalFormat);
		GLenum data_type = static_cast<GLenum>(mPixelDataFormat);
		Bind();
		glTexSubImage2D(target, 0, x, y, width, height, in_form, data_type, buffer);
		UnBind();
	}

}



#endif // !_GL_TEXTURE_
