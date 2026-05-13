# CHANGELOG

## v1.4.0 (2026-05-13)

### Bug fixes

* **notebook**: update initialization with new parameters `accelOffset` and `accelConversion` ([`8df0afa`](https://github.com/ibressler/dobot1-control/commit/8df0afa9b4114d29015783aca00d8eb50b360843))

* **calibration**: show help message and exit when no arguments are provided ([`0e9792e`](https://github.com/ibressler/dobot1-control/commit/0e9792ee337f8c32976bac7f7d736268b503c278))

* **calibration**: improve device port detection for cross-platform support ([`1c47b3a`](https://github.com/ibressler/dobot1-control/commit/1c47b3a9335e2d498ec758662785e4daa4ffa6a2))

* **DobotDriver**: replace sca1000Sensors parameter by free accelConversion ([`340f967`](https://github.com/ibressler/dobot1-control/commit/340f967b5c1eb69b23cd1a75e8ffec20d3536e9e))

* **DobotDriver**: replace math constant ([`a283dc8`](https://github.com/ibressler/dobot1-control/commit/a283dc86da8b9044890d938858507dd89352b89e))

* **DobotSDK**: refine slice skipping logic ([`d87df6b`](https://github.com/ibressler/dobot1-control/commit/d87df6b5fab2dac30b8de60a9e369315d027338a))

* **DobotSDK**: joint limits for Dobot with SCA1000 accelerometers ([`1209fac`](https://github.com/ibressler/dobot1-control/commit/1209facbcc26ef40b82ec8dbc74c6aacc1a515fe))

* **DobotSDK**: clip joint velocity and acceleration limits to feasible ranges ([`a5673a6`](https://github.com/ibressler/dobot1-control/commit/a5673a6c016d99816be29ff341969663f96b1219))

* **notebook**: adjust to updated constructor arguments ([`9d003ea`](https://github.com/ibressler/dobot1-control/commit/9d003ea2378fc87918728bca5a1f0a8f440e69c0))

* **DobotSDK**: suppress debug output when disabled ([`f62e318`](https://github.com/ibressler/dobot1-control/commit/f62e318ca78d44c6cfd9300fc69ed552e9ae210e))

* **notebook**: adapt Dobot init() for SCA1000 sensors and end effector offsets ([`08d65d4`](https://github.com/ibressler/dobot1-control/commit/08d65d43c3d4a1d3ab5d8829bb7c6300c1ed59fb))

* **DobotSDK**: check for empty data arrays to prevent runtime errors in plotting ([`d994972`](https://github.com/ibressler/dobot1-control/commit/d9949720788ac9e47b380d3c221f7e9e7d1dded7))

* **calibrate-accelerometers**: add units (°, mm) to printed outputs for clarity ([`204d3ff`](https://github.com/ibressler/dobot1-control/commit/204d3ff9b0b4a19c36d53e569422dbac9039d40e))

* **calibrate-accel**: glob serial device name on linux ([`70de135`](https://github.com/ibressler/dobot1-control/commit/70de135970f74f61dd10e6b4445f1a6fa82c8d82))

* **SegmentParams._solve_common**: mismatch factor calculation ([`5800f37`](https://github.com/ibressler/dobot1-control/commit/5800f371b46a9bc18b701066b82daba83fb6c5a0))

* **Dobot._prepareAnglesSlice**: order of step arithmetic for possibly negative step diffs ([`0b59af4`](https://github.com/ibressler/dobot1-control/commit/0b59af4acdf21150785e88dc04750d1f181eae4d))

* **DobotSDK.MoveWithSpeed**: skip calculation for zero distance, prevent div-by-zero err ([`f5721a3`](https://github.com/ibressler/dobot1-control/commit/f5721a3530764cb2bc0a097da7e2f19963381f55))

### Code style

* **notebook**: disable debug output, fix comments and formatting ([`88c4bec`](https://github.com/ibressler/dobot1-control/commit/88c4becd7c0f6a7af9ee6e99da37d43dee40a56a))

* **DobotSDK**: improve comment clarity ([`d5acf75`](https://github.com/ibressler/dobot1-control/commit/d5acf75f5d3409781427ecd747194ccc4431ed5b))

* **DobotSDK, DobotKinematics**: simplify debug methods by unifying print logic ([`a326e8d`](https://github.com/ibressler/dobot1-control/commit/a326e8d01cdcce5535dbd24b0aff61e2593a98a7))

* **DobotKinematics**: simplify debug method with improved string formatting ([`cce40d4`](https://github.com/ibressler/dobot1-control/commit/cce40d4b65db9668fd41e4a7542ae785d335271c))

* **DobotKinematics**: fix spelling, improve formatting, and add static methods ([`d876f09`](https://github.com/ibressler/dobot1-control/commit/d876f09c372a602637f22eba7c06b8e7dd111f1d))

* **DobotKinematics**: indenting: tabs to spaces ([`ad4eb26`](https://github.com/ibressler/dobot1-control/commit/ad4eb267ba13d0cc1b26c0b0a7d47c95d74fbd82))

* **DobotDriver**: black formatting consistency ([`47acbe9`](https://github.com/ibressler/dobot1-control/commit/47acbe99605c530ad2866d847f42388dbbba5e72))

* **DobotSDK**: improve string formatting consistency and minor grammar fix ([`faf4f66`](https://github.com/ibressler/dobot1-control/commit/faf4f66c6658e416dd49d600dfb906aaf22afd17))

* **DobotDriver+SDK**: convert indents to spaces ([`2b4618a`](https://github.com/ibressler/dobot1-control/commit/2b4618ab9c91b5776af9772a89f67a5338c84325))

* **DobotDriver**: consistent formatting and minor spelling corrections ([`9d067e2`](https://github.com/ibressler/dobot1-control/commit/9d067e2b443103be99cee3d9790d5f40f97779bc))

* style: spelling ([`e63e525`](https://github.com/ibressler/dobot1-control/commit/e63e525b919e53e2ffe03182b112f885ada4f80f))

### Continuous integration

* ci: publish job with pypi environment ([`07ac694`](https://github.com/ibressler/dobot1-control/commit/07ac694938110e58ae57852580ae6d5a57557b56))

* ci: set Python version to 3.14 ([`d333a81`](https://github.com/ibressler/dobot1-control/commit/d333a81e7c46617564b1252745032215e98b2e2f))

### Documentation

* **Project**: update image URLs, improve instructions, and refine templates ([`faa7d6d`](https://github.com/ibressler/dobot1-control/commit/faa7d6d412e81b08e940a527318ab9719775660d))

* **LICENSE**: minor formatting ([`608f279`](https://github.com/ibressler/dobot1-control/commit/608f2791abc86797725d6b9aa0eaee72f563e88f))

* **README**: Update from project template ([`651e1b6`](https://github.com/ibressler/dobot1-control/commit/651e1b6b7fdd9c0f6187bbd39f08e073ee264b69))

* **license**: add MIT License and update copyright information in README ([`4456257`](https://github.com/ibressler/dobot1-control/commit/4456257338dd0e6ab8fb40d406185032e2d8d01e))

* **README**: clarify source and firmware requirement ([`1c815d4`](https://github.com/ibressler/dobot1-control/commit/1c815d429b432d2d3aa1c8d3d608e571daf009cb))

* **README**: add detailed instructions for enabling accelerometer reporting mode ([`3ef1c26`](https://github.com/ibressler/dobot1-control/commit/3ef1c26bc8ddb2b16c6ae6a12827c81078644ad6))

* **README**: adjust circle path image size ([`ecd5a2a`](https://github.com/ibressler/dobot1-control/commit/ecd5a2a8c37685bfa7f0daf190633262f6c7e331))

* **README**: update with motion planning, accelerometer calibration, and usage details ([`7a32c79`](https://github.com/ibressler/dobot1-control/commit/7a32c791bdddfb821f826b76cdaca750a805a5f1))

* **README**: typo ([`6daf0fb`](https://github.com/ibressler/dobot1-control/commit/6daf0fb869c4309bb0a2df5e3d6d09c335a2671f))

* **README**: fix broken links to maxosprojects repository ([`1248840`](https://github.com/ibressler/dobot1-control/commit/12488400750bb13096b1950dd36e093516b86595))

* **README**: fix broken links to maxosprojects repository ([`6af60d8`](https://github.com/ibressler/dobot1-control/commit/6af60d8d4bab6baf2315348e340cef33030ce0b6))

* **README**: fix broken links to maxosprojects repository ([`77e1169`](https://github.com/ibressler/dobot1-control/commit/77e1169a67eaece3aa4516dd277f2fe28f47ea0f))

* docs: add initial README for Dobot v1 Control ([`620cfee`](https://github.com/ibressler/dobot1-control/commit/620cfee31100efea0b40ebc3d0373344c5db53c4))

* docs: author list updated ([`ced77dd`](https://github.com/ibressler/dobot1-control/commit/ced77dd697f5e2a455e3ae7b725425ef6485a01f))

* **notebook**: comments added debug output disabled ([`1acbd7e`](https://github.com/ibressler/dobot1-control/commit/1acbd7e1a27b9e37bed2c4319ce1ad40452ea05e))

* docs: improve grammar, consistency, and clarity in comments and docstrings ([`de05319`](https://github.com/ibressler/dobot1-control/commit/de053192dd1189359c5179baf1abb038ecdb3660))

* **calibrate-accelerometers**: improve instructions formatting and grammar adjustments ([`01bed53`](https://github.com/ibressler/dobot1-control/commit/01bed533233abd07a6432eefb82cf4801b8e909f))

* **DobotDriver**: fix and add docstrings for all methods and their parameters ([`27112ca`](https://github.com/ibressler/dobot1-control/commit/27112cab3f34b09ddd7c3093319270fdee11b009))

* **DobotDriver**: add comprehensive docstring for class initialization parameters ([`f53659f`](https://github.com/ibressler/dobot1-control/commit/f53659f906e1593cf12604d5238a6f7bad3a502f))

* **DobotSDK.__init__**: add detailed docstring for class initialization parameters ([`d2399eb`](https://github.com/ibressler/dobot1-control/commit/d2399eb18c153a251e7815403cb9a73560733257))

### Features

* **calibration**: add CLI option for end effector offset ([`dc9e38b`](https://github.com/ibressler/dobot1-control/commit/dc9e38b4309394ba7711a4f6d0ed6265ab7477a0))

* **calibration**: refine accelerometer calibration with detailed offsets and conversion calculation ([`8ea7f7b`](https://github.com/ibressler/dobot1-control/commit/8ea7f7bf693af022f1950ba484a914155daf068d))

* **calibration**: add CLI options for first and second calibration positions ([`0c16710`](https://github.com/ibressler/dobot1-control/commit/0c16710fca22e02030606a65076f3b34d173cc15))

* **calibration**: add positions mode ([`0742986`](https://github.com/ibressler/dobot1-control/commit/0742986ae957de54ba11cf38de7aacc732922085))

* **DobotSDK**: add accelOffset parameter and pass to driver for accelerometer configuration ([`f6a3bf9`](https://github.com/ibressler/dobot1-control/commit/f6a3bf96326202e464f9e2389f6afb298dbf2554))

* **DobotKinematics**: add unit tests for kinematics and improve documentation ([`3553123`](https://github.com/ibressler/dobot1-control/commit/3553123e698b37ed912ef6533f8819e2e948b51f))

* **DobotSDK**: add posAngles property for joint angle position ([`daa8585`](https://github.com/ibressler/dobot1-control/commit/daa85855053f000d4d324a96f62fda5eb5a51fe2))

* **DobotSDK**: add static method to format positional data ([`8cbfabd`](https://github.com/ibressler/dobot1-control/commit/8cbfabd67b9640f1dbc3eddc5ac39f6aae9db567))

* **DobotSDK**: introduce configurable angular limits for joints ([`4f041f6`](https://github.com/ibressler/dobot1-control/commit/4f041f60c745a7a59d37b323e4e75732e10bc3ef))

* **DobotSDK**: add configurable joint velocity and acceleration limits with updated docstrings ([`2c25e71`](https://github.com/ibressler/dobot1-control/commit/2c25e716887b9fbad3419ce08a31f5b9888cec9a))

* **DobotKinematics, DobotSDK**: add configurable end effector offset with updated docstrings ([`b3cf044`](https://github.com/ibressler/dobot1-control/commit/b3cf0446697f08e98bdf44437ccdbc4649c4649f))

* **DobotDriver**: add support for SCA1000-D01 sensors with configurable accel conversion ([`bc377b4`](https://github.com/ibressler/dobot1-control/commit/bc377b4b62048028b298370fc7f2cadaa5ec3428))

* **calibrate-accel**: limit sensor queries to 4 per seconds ([`53174d4`](https://github.com/ibressler/dobot1-control/commit/53174d491f21b21a129ecdddc2060d4bb4eea2b4))

* **notebook**: jupyter notebook for testing SDK and movements semi-interactively ([`628725e`](https://github.com/ibressler/dobot1-control/commit/628725ef10009ff68a932a49203624ce300fd9a8))

* **DobotSDK**: omit debug output in non-debug mode ([`fad4182`](https://github.com/ibressler/dobot1-control/commit/fad418259b1bd7be6fc14769248ad91933a30580))

* **DobotSDK**: Dobot.pos remembers own absolute coordinates ([`1c73253`](https://github.com/ibressler/dobot1-control/commit/1c73253157b87b5854eac137c37d7239209924a6))

### Refactoring

* **docs/codebase**: standardize author/license fields, remove unused docs ([`abbd27c`](https://github.com/ibressler/dobot1-control/commit/abbd27c194f86fb4abb2e4fe335da8a9e0e3ae68))

* **Codebase**: align formatting to PEP8, improve readability ([`934f6de`](https://github.com/ibressler/dobot1-control/commit/934f6de05423c9f271d82b47c64dbec205304abd))

* **Project**: file structure standardized, tool config added ([`2096409`](https://github.com/ibressler/dobot1-control/commit/2096409787e98d9a4563b1e73874deddec5eb445))

* **Dobot**: remove joint angle limits, revert 4f041f60 ([`f111647`](https://github.com/ibressler/dobot1-control/commit/f11164778c8ef561477c127df23cd580adb8f094))

* **DobotSDK**: move valueToStr and arrayToStr to DobotBase ([`03c74eb`](https://github.com/ibressler/dobot1-control/commit/03c74eb2ce0e65e4f2cd9402d2554c38cbebb137))

* **DobotSDK**: move accel offset to driver ([`4c1bb09`](https://github.com/ibressler/dobot1-control/commit/4c1bb099aff593d6654c4a66253759a4c6fa1048))

* **DobotBase**: centralize axes/joint names for consistency ([`08bdbf0`](https://github.com/ibressler/dobot1-control/commit/08bdbf04cda3c6119fdf0d229196707957ae3b3c))

* **DobotSDK**: remove unused code; fix debug output ([`8fe79a2`](https://github.com/ibressler/dobot1-control/commit/8fe79a21089528b3d44c9f18d8a000b35d014a96))

* **DobotKinematics**: return NumPy arrays ([`3f23766`](https://github.com/ibressler/dobot1-control/commit/3f23766d1b57df2d143f11cd8cf3a131b336c865))

* **DobotSDK**: move utility functions for array debugging and printing ([`f6525cd`](https://github.com/ibressler/dobot1-control/commit/f6525cd3e6467b9ee2eda80e99038aa5914319f4))

* **DobotSDK, DobotKinematics, DobotDriver**: introduce to centralize debugging functionality ([`e6adbc6`](https://github.com/ibressler/dobot1-control/commit/e6adbc6ff3b5d209532c0fa83d82500248148f37))

* **DobotKinematics**: remove unused function and constants ([`053f192`](https://github.com/ibressler/dobot1-control/commit/053f192fcf610ce97403686f0509d7822a9fa20c))

* **DobotDriver**: remove accel*ToAngle() ([`ca090ea`](https://github.com/ibressler/dobot1-control/commit/ca090ea211e8435150fc968a00f795b8a452e8a3))

* **Dobot.InitializeAccelerometers**: new _get_accelerometers_raw() used by fpga & non-fpga variant ([`5a0ca9a`](https://github.com/ibressler/dobot1-control/commit/5a0ca9a27925497242d1f3c017d3607dfc116f58))

* **DobotSDK._moveToAnglesSlice**: enable skipping of zero-step slices ([`cece309`](https://github.com/ibressler/dobot1-control/commit/cece309ec4bf6f6fcd0d7bdc7f137c90798ec862))

* **DobotSDK.MoveWithSpeed**: precise motion planning and velocity synchronization across multiple segments ([`a6faae2`](https://github.com/ibressler/dobot1-control/commit/a6faae202aa5d1946d9ca8cd30adff4991ab27b5))

* **DobotSDK.MoveWithSpeed**: synchronize joint-indiv. acceleration and deceleration phases for multiple segments ([`4a5b237`](https://github.com/ibressler/dobot1-control/commit/4a5b237276219b5055d5bd6d8d09e566fbaf37aa))

* **DobotDriver**: replace `exit` with `sys.exit` for clearer intent and consistency ([`e60c006`](https://github.com/ibressler/dobot1-control/commit/e60c006ca3fa0dd4fab440ec1a3e83e0238e34e0))

* **DobotSDK.MoveWithSpeed**: optimize motion planning with lookahead waypoint speed profiling ([`89be1df`](https://github.com/ibressler/dobot1-control/commit/89be1df3383767eb8cc783fdbdf46423982bca5f))

* **DobotKinematics.anglesFromCoordinates**: add debug flag and expect vector input ([`13ab38b`](https://github.com/ibressler/dobot1-control/commit/13ab38b861d6824fa9774a7b609d9afafa9c2ce0))

* **DobotSDK.MoveWithSpeed**: vectorize target position ([`92afda0`](https://github.com/ibressler/dobot1-control/commit/92afda0584b0453a03915c0dc7d9cc48f7a9e2d4))

* **notebook**: remove lower-level driver control, adjust high-level control ([`aa0e18a`](https://github.com/ibressler/dobot1-control/commit/aa0e18ac73ec7863ad1bba6cf2531426bd77761f))

* **DobotSDK, DobotDriver**: streamline joint commands and plotting logic with vectorized data structures ([`3141874`](https://github.com/ibressler/dobot1-control/commit/31418743acb96a93674a901d0dad61b90b0ebc00))

* **DobotSDK.MoveWithSpeed**: vectorize step calculations using NumPy for cleaner logic ([`88d02f8`](https://github.com/ibressler/dobot1-control/commit/88d02f8d7ea817b09fd7c41488ef006cd1bc5114))

* **notebook**: comment on device permissions and usage ([`0fb9e0c`](https://github.com/ibressler/dobot1-control/commit/0fb9e0cf479d5d0eb31e19b859ee9f3d55ac4764))

* **DobotSDK**: consolidate coordinate tracking logic and enhance plotting with slice data visualization ([`e783709`](https://github.com/ibressler/dobot1-control/commit/e7837094a7eef90d68b8af99b4f93408dd37205b))

* **DobotSDK**: introduce DobotPlotter class for enhanced plotting and clean up legacy code ([`84b5df4`](https://github.com/ibressler/dobot1-control/commit/84b5df4ec2c9e80fa3a4a1623e068cbea7c23636))

* **DobotSDK**: clean up unused methods, improve formatting, and fix minor issues ([`55d3983`](https://github.com/ibressler/dobot1-control/commit/55d39838367fea37c4e8f0c13f9a80b4a8134b4e))

* **DobotDriver**: update timeout parameter and remove unused methods ([`da627a6`](https://github.com/ibressler/dobot1-control/commit/da627a673c93f59f63a17513bc55008b7e07674f))

### Unknown Scope

* let git ignore python temp files ([`749f87a`](https://github.com/ibressler/dobot1-control/commit/749f87a0c46c100456f71a09eb841b467cf3b016))

## v1.3.0 (2026-03-31)

### Unknown Scope

* initial import of original code from maxosprojects/open-dobot ([`357d3f8`](https://github.com/ibressler/dobot1-control/commit/357d3f8d4085d169738f6c8ac42b00f5e65a0b04))
