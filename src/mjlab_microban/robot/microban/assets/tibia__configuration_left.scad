% scale(1000) import("tibia__configuration_left.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:
// cube([10, 10, 10], center=true);
// cylinder(r=10, h=10, center=true);
// sphere(10);

translate([17.2-37, -9, 26.25])
cube([2.5, 18, 23], center=true);

translate([17.25-37, -5.8, 15.27])
rotate([45, 0, 0])
cube([2.5, 18, 16.5], center=true);

translate([17.25-37, -4.66, 36.08])
rotate([-45, 0, 0])
cube([2.5, 21.213, 16.5], center=true);

translate([17.25-37, 0, 9.5])
rotate([0, 90, 0])
cylinder(r=9, h=2.5, center=true);


translate([-17.25-37, -9, 26.25])
cube([2.5, 18, 23], center=true);

translate([-17.25-37, -5.8, 15.27])
rotate([45, 0, 0])
cube([2.5, 18, 16.5], center=true);

translate([-17.25-37, -4.66, 36.08])
rotate([-45, 0, 0])
cube([2.5, 21.213, 16.5], center=true);

translate([-17.25-37, 0, 9.5])
rotate([0, 90, 0])
cylinder(r=9, h=2.5, center=true);


translate([0-37, 8, 48.713])
cube([37, 25, 5], center=true);

translate([0-37, 8, 65.7])
cube([23, 20, 29], center=true);

translate([0-37, -3.7, 47.09])
rotate([-45, 0, 0])
cube([37, 7, 4.659], center=true);
