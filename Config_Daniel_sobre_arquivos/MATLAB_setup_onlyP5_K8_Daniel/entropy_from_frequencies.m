% ## Copyright (C) 2019 MateusMarcuzzo
% ## 
% ## This program is free software: you can redistribute it and/or modify it
% ## under the terms of the GNU General Public License as published by
% ## the Free Software Foundation, either version 3 of the License, or
% ## (at your option) any later version.
% ## 
% ## This program is distributed in the hope that it will be useful, but
% ## WITHOUT ANY WARRANTY; without even the implied warranty of
% ## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% ## GNU General Public License for more details.
% ## 
% ## You should have received a copy of the GNU General Public License
% ## along with this program.  If not, see
% ## <https://www.gnu.org/licenses/>.
% 
% ## -*- texinfo -*- 
% ## @deftypefn {} {@var{retval} =} entropy (@var{input1}, @var{input2})
% ##
% ## @seealso{}
% ## @end deftypefn
% 
% ## Author: MateusMarcuzzo <MateusMarcuzzo@PC-MATEUS>
% ## Created: 2019-03-27

function h = entropy_from_frequencies(ffq, log_base)
%   FFQ must be already in frequency form, integer values are not count


% IMPORTANT: THE FREQUENCIES MUST BE COLUMN-WISE
% I.E assert(all(sum(ffq,1) == 1))


% since it's double values and we have float problems, we will assert the way below.
assert(all(sum(ffq,1) >= 0.95));
assert(all(sum(ffq,1) <= 1.1));


h=-sum(ffq.*log2(ffq+eps),1)./log2(log_base);

end
