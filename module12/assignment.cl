//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void square(__global * buffer, const unsigned int size )
{
	size_t id = get_global_id(0);
	int avg = 0;
	for (unsigned int i = id; i < id + size; ++i)
	{
		avg += buffer[i];
		buffer[i] = 0;
	}
	buffer[id] = avg / size;
}