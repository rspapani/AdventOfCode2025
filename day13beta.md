# Day X: Ice Cavern Mining

The elves have discovered a massive ice cavern and invited you along on an ice mining expedition. When you arrive at the site, you're greeted by a breathtaking sight: a crystalline ice formation stretching deep into the frozen ground, layer upon layer disappearing into darkness below.

## Part 1: Exposed Ice Assessment

"We need to know how much ice we can extract right away," explains the head mining engineer, pulling out a strange scanning device. "We can only mine blocks that are _exposed_—anything deeper requires more careful extraction."

The device produces a 3D map of the entire cavern structure, showing which blocks are solid ice (`#`) and which are air pockets (`.`). The cavern is represented as layers descending from the surface, with layer 0 at the top.

An ice block is considered **exposed** if at least one of its faces (up, down, left, right, front, or back—no diagonal connections) is adjacent to **external air**. External air means any air that's connected to the open air at the surface (layer 0) through a continuous path of air blocks using only orthogonal adjacency (no diagonals).

The cavern is surrounded by a thick ice shell—all blocks on the outer edges are guaranteed to be solid ice, except for openings on the surface layer where external air can enter.

**Your task:** Count how many ice blocks are exposed to external air.

### Example

Consider this small ice cavern (12×12×12 blocks, shown layer by layer from surface downward):

```
############
############
####.##..###
########..##
############
###.#.######
########.###
############
#####.######
#####.######
############
############

############
############
####.##..###
########..##
############
###.#.######
########.###
############
#####.######
#####.######
############
############

############
############
####.##..###
########..##
############
###.#.######
########.###
############
#####.######
#####.######
############
############

############
############
#######.####
########..##
############
#####.######
########.###
############
#####.######
#####.######
############
############

############
############
############
########.###
############
###.########
########.###
############
############
#####.######
############
############

############
############
############
##...#######
#.....######
#.....######
#.....######
##...#######
############
############
############
############

############
############
#######..###
#.....#..###
#.....######
#.....######
#.....######
#.....######
########.###
############
############
############

############
############
###.###..###
#.....#..###
#.....######
#......#####
#.....#...##
#..........#
###.##.....#
######.....#
#######...##
############

############
############
############
#.....######
#.....######
#.....######
#..........#
#..........#
######.....#
######.....#
######.....#
############

############
############
############
##...#######
#.....######
#.....##.###
#..........#
##...#.....#
#####......#
######.....#
######.....#
############

############
############
############
############
############
###.########
######.....#
######.....#
######.....#
######.....#
######.....#
############

############
############
############
############
############
############
############
############
############
############
############
############
```

In this example:

- Surface air enters through openings on layer 0
- External air flows down through connected air blocks
- Ice blocks adjacent to any external air are considered exposed

The answer for the example is `128`.

### Puzzle Input

Your puzzle input is a 3D scan from the mining device, showing the cavern layer by layer. Each layer is a 2D grid where:

- `#` represents solid ice
- `.` represents air

Layers are separated by blank lines and descend from the surface (layer 0 at the top) downward.

**Sample input:** See the example above

**How many ice blocks are exposed to external air?**

---

## Part 2: Structural Integrity Analysis

The elves gather around, looking concerned. "Before we start mining," the engineer says, "we need to assess the structural integrity. Removing ice creates instability, but the real danger comes from _trapped air pockets_ inside the ice."

"Trapped air pockets?" you ask.

"Air that's completely sealed inside the ice, with no connection to the outside. These pockets create weak points. If we mine away the wrong ice blocks, the whole structure could collapse. We need to know the total volume of trapped air so we can calculate how much reinforcement we'll need."

**Your task:** Count how many air blocks are trapped inside the ice—that is, air blocks that are _not_ connected to the external air at the surface.

### Example

Using the same cavern as before, we can now identify trapped air pockets:

```
############
############
####.##..###
########..##
############
###.#.######
########.###
############
#####.######
#####.######
############
############

############
############
####.##..###
########..##
############
###.#.######
########.###
############
#####.######
#####.######
############
############

############
############
####.##..###
########..##
############
###.#.######
########.###
############
#####.######
#####.######
############
############

############
############
#######.####
########..##
############
#####.######
########.###
############
#####.######
#####.######
############
############

############
############
############
########.###
############
###.########
########.###
############
############
#####.######
############
############

############
############
############
##...#######
#.....######
#.....######
#.....######
##...#######
############
############
############
############

############
############
#######..###
#.....#..###
#.....######
#.....######
#.....######
#.....######
########.###
############
############
############

############
############
###.###..###
#.....#..###
#.....######
#......#####
#.....#...##
#..........#
###.##.....#
######.....#
#######...##
############

############
############
############
#.....######
#.....######
#.....######
#..........#
#..........#
######.....#
######.....#
######.....#
############

############
############
############
##...#######
#.....######
#.....##.###
#..........#
##...#.....#
#####......#
######.....#
######.....#
############

############
############
############
############
############
###.########
######.....#
######.....#
######.....#
######.....#
######.....#
############

############
############
############
############
############
############
############
############
############
############
############
############
```

Analysis:

- External air enters through surface openings and flows downward
- Most air pockets are connected to the surface
- However, there are sealed air pockets completely surrounded by ice

The answer for the example is `229`.

**How many air blocks are trapped inside the ice?**
