#include <HostLink.h>

int main()
{
  HostLink hostLink;
  hostLink.boot("code.v", "data.v");
  hostLink.dump();
  return 0;
}
