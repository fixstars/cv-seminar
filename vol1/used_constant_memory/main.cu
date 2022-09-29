/*
Copyright 2022 Fixstars Corporation
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http ://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <opencv2/cudev.hpp> // NG
//#include <opencv2/cudev/common.hpp> //OK
#include <iostream>

__constant__ float buffer[16384]; // 64KB((64*1024)/4)

int main(int argc, char *argv[])
{
    std::exit(EXIT_SUCCESS);
}
