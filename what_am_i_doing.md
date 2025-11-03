here, i am just trying to write stuff as i do them, so that i can remember it later

so, the idea is we need to be able to do multi threading,
and for that we'll be using OpenMP -- found some slides. attaching here for reference -- https://engineering.purdue.edu/~smidkiff/ece563/files/ECE563OpenMPTutorial.pdf

first lets add components for thread management
we need the number of threads that we'll be using, and a unique scratch space for all the threads

when a unique_ptr goes out of scope, the object owned by it is automatically deleted

in c++, when we use unique_ptr, it ensures exclusive ownership of the dynamically allocated object, and thus prevents memory leaks. it is possible to change ownership from this pointer to something else too.

std::make_unique<ScratchSpace>()

This is a C++14 factory function that does two things:

1. Calls new ScratchSpace()

2. Wraps the raw pointer in a std::unique_ptr<ScratchSpace>