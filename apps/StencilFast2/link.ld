

















































































OUTPUT_ARCH( "riscv" )
MEMORY
{
  instrs : ORIGIN = 134217728 +1024, LENGTH = ((4 << 13) - 1024)
  globals : ORIGIN = (134217728 + (4 << 13)), LENGTH = 1 << 20
}
SECTIONS
{
  .text : { *.o(.text*) } > instrs
  .bss : { *.o(.bss*) } > globals = 0
  .rodata : { *.o(.rodata*) } > globals
  .sdata : { *.o(.sdata*) } > globals
  .data : { *.o(.data*) } > globals
  .captable : { *.o(.captable*) } > globals
  __cap_relocs : { *.o(__cap_relocs*) } > globals
  .eh_frame_hdr : ONLY_IF_RW { KEEP (*(.eh_frame_hdr))
                                     *(.eh_frame_hdr.*) } > globals
  .eh_frame : ONLY_IF_RW { KEEP (*(.eh_frame)) *(.eh_frame.*) } > globals
}
